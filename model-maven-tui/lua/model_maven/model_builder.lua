local M = {}
local label = require('model_maven.locale').text
local api = vim.api
local function unique_name(rows, stem)
  local names = {}
  for _, row in ipairs(rows) do names[row.name] = true end
  local n = 1
  while names[stem .. n] do n = n + 1 end
  return stem .. n
end

local function valid_name(value)
  if #value <= 48 and (value:match('^%a[%w_]*$') or (value:find('[\128-\255]')
      and not value:find('[%c%s:;=@,]') and value:sub(1,1) ~= '_')) then return value end
end

local function parse_shape(raw)
  local shape, dynamic = {}, {}
  local dims = vim.split(raw, ',', { plain = true })
  if #dims > 8 then return end
  for i, value in ipairs(dims) do
    value = vim.trim(value)
    if not value:match('^%??%d+$') then return end
    local n = tonumber(value:gsub('^%?', ''), 10)
    if not n or n < 1 or n > 1000000 then return end
    shape[i] = n
    if value:sub(1, 1) == '?' then dynamic[#dynamic + 1] = i - 1 end
    dims[i] = value
  end
  return shape, dynamic, table.concat(dims, ',')
end

local function new_input(name, shape) return { name = name, shape = shape, dtype = '' } end
local function new_output(name) return { name = name, width = '1', loss = 'mse', weight = '1' } end

local layer_names = { linear = 'Linear', relu = 'ReLU', gelu = 'GELU', layer_norm = 'LayerNorm', dropout = 'Dropout', factory = 'Custom', residual = 'Residual', identity = 'Identity', parallel = 'Parallel' }
local function branch_groups(layer)
  if layer.type == 'residual' then return {layer.branch} end
  if layer.type == 'parallel' then return layer.branches end
  return {}
end
local function supported_layer(layer, depth)
  depth = depth or 0
  if type(layer) ~= 'table' or not layer_names[layer.type] then return false end
  if layer.id ~= nil and (type(layer.id) ~= 'string' or #layer.id ~= 34 or not layer.id:match('^n_[0-9a-f]+$')) then return false end
  if layer.type == 'residual' then
    if depth >= 8 or type(layer.branch) ~= 'table' or not vim.islist(layer.branch) or #layer.branch < 1 or #layer.branch > 16 then return false end
    for key in pairs(layer) do if key ~= 'type' and key ~= 'id' and key ~= 'branch' then return false end end
    for _, item in ipairs(layer.branch) do if not supported_layer(item, depth + 1) then return false end end
    return true
  end
  if layer.type == 'parallel' then
    if depth >= 8 or type(layer.branches) ~= 'table' or not vim.islist(layer.branches)
        or #layer.branches < 2 or #layer.branches > 8 or (layer.merge ~= 'add' and layer.merge ~= 'concat') then return false end
    for key in pairs(layer) do if key ~= 'type' and key ~= 'id' and key ~= 'branches' and key ~= 'merge' then return false end end
    for _, group in ipairs(layer.branches) do
      if type(group) ~= 'table' or not vim.islist(group) or #group < 1 or #group > 16 then return false end
      for _, item in ipairs(group) do if not supported_layer(item, depth + 1) then return false end end
    end
    return true
  end
  if layer.type == 'factory' then
    for key in pairs(layer) do if key ~= 'type' and key ~= 'id' and key ~= 'reference' and key ~= 'output_width' and key ~= 'params' then return false end end
    return type(layer.reference) == 'string' and #layer.reference <= 240
      and layer.reference:match('^[^%s:;=]+:[^%s:;=]+$') ~= nil
      and type(layer.output_width) == 'number' and layer.output_width % 1 == 0
      and layer.output_width >= 1 and layer.output_width <= 65536
      and (layer.params == nil or (type(layer.params) == 'table' and (not next(layer.params) or not vim.islist(layer.params))))
  end
  local field = layer.type == 'linear' and 'features' or layer.type == 'dropout' and 'probability'
  for key in pairs(layer) do if key ~= 'type' and key ~= 'id' and key ~= field then return false end end
  if layer.type == 'linear' then
    return type(layer.features) == 'number' and layer.features % 1 == 0 and layer.features >= 1 and layer.features <= 65536
  end
  if layer.type == 'dropout' then
    return type(layer.probability) == 'number' and layer.probability >= 0 and layer.probability < 1
  end
  return true
end

local function count_layers(layers)
  local count = 0
  for _, layer in ipairs(layers or {}) do
    count = count + 1
    for _, group in ipairs(branch_groups(layer)) do count = count + count_layers(group) end
  end
  return count
end

local function visit_layers(layers, visit)
  for _, layer in ipairs(layers or {}) do
    visit(layer)
    for _, group in ipairs(branch_groups(layer)) do visit_layers(group, visit) end
  end
end

local function unique_layer_ids(inputs)
  local seen, unique = {}, true
  for _, input in ipairs(inputs or {}) do
    visit_layers(input.layers, function(layer)
      if layer.id then
        if seen[layer.id] then unique = false else seen[layer.id] = true end
      end
    end)
  end
  return unique
end

local function estimated_width(layer, input_width)
  if not input_width then return nil end
  if layer.type == 'linear' then return layer.features end
  if layer.type == 'factory' then return layer.output_width end
  local groups = branch_groups(layer)
  if #groups == 0 then return input_width end
  local widths = {}
  for _, group in ipairs(groups) do
    local width = input_width
    for _, child in ipairs(group) do width = estimated_width(child, width) end
    if not width then return nil end
    widths[#widths + 1] = width
  end
  if layer.type == 'residual' then return widths[1] == input_width and input_width or nil end
  if layer.merge == 'add' then
    for _, width in ipairs(widths) do if width ~= widths[1] then return nil end end
    return widths[1]
  end
  local total = 0
  for _, width in ipairs(widths) do total = total + width end
  return total <= 1000000 and total or nil
end

function M.load(project)
  local path = project .. '/model-maven.json'
  local stat, err, code = vim.uv.fs_lstat(path)
  if not stat then return nil, code ~= 'ENOENT' and err or nil end
  if stat.type ~= 'file' or stat.size > 128 * 1024 then return nil, 'model-maven.json must be a regular file up to 128 KiB.' end
  local file, open_error = io.open(path, 'rb')
  if not file then return nil, open_error end
  local bytes = file:read(128 * 1024 + 1)
  file:close()
  local ok, document = pcall(vim.json.decode, bytes)
  -- The backend treats explicit null as absent only on optional fields.
  -- Preserve every other field so unsupported declarations are still refused.
  if ok and type(document) == 'table' then
    for _, key in ipairs({'training', 'supervision'}) do
      if document[key] == vim.NIL then document[key] = nil end
    end
    if document.kind == 'image_classification' then
      for _, key in ipairs({'source', 'runtime'}) do if document[key] == vim.NIL then document[key] = nil end end
    elseif document.data == vim.NIL then document.data = nil end
    if type(document.data) == 'table' and document.data.binding == vim.NIL then document.data.binding = nil end
  end
  local function fields(row, allowed)
    if type(row) ~= 'table' then return false end
    for field in pairs(row) do if not allowed[field] then return false end end
    return true
  end
  local invalid = 'Unsupported or invalid model-maven.json.'
  if ok and type(document) == 'table' and document.version == 1 and document.kind == 'image_classification' then
    if not fields(document, { version = true, kind = true, preset = true, data = true, classes = true, training = true, supervision = true, source = true, runtime = true })
        or type(document.preset) ~= 'string' or type(document.data) ~= 'string'
        or not vim.islist(document.classes) then return nil, invalid end
    for _, name in ipairs(document.classes) do if type(name) ~= 'string' then return nil, invalid end end
    local training, supervision = document.training, document.supervision
    if training ~= nil and (not fields(training, {mode=true, entry=true, params=true, optimizer=true})
        or type(training.mode) ~= 'string' or type(training.entry) ~= 'string'
        or (training.params ~= nil and type(training.params) ~= 'table')
        or (training.optimizer ~= nil and training.optimizer ~= vim.NIL
          and not fields(training.optimizer, {factory=true, params=true, lr=true, weight_decay=true}))) then return nil, invalid end
    if supervision ~= nil and (not fields(supervision, {file=true, sha256=true}) or type(supervision.file) ~= 'string'
        or type(supervision.sha256) ~= 'string' or #supervision.sha256 ~= 64 or supervision.sha256:find('[^0-9a-f]')) then return nil, invalid end
    local source = document.source
    if source ~= nil and (not fields(source, {file=true, sha256=true}) or type(source.file) ~= 'string'
        or type(source.sha256) ~= 'string' or #source.sha256 ~= 64 or source.sha256:find('[^0-9a-f]')) then return nil, invalid end
    local runtime = document.runtime
    if runtime ~= nil and (not fields(runtime, {project_import_roots=true, working_directory=true}) or (runtime.project_import_roots ~= nil
        and (type(runtime.project_import_roots) ~= 'table' or not vim.islist(runtime.project_import_roots)))
        or (runtime.working_directory ~= nil and runtime.working_directory ~= vim.NIL and type(runtime.working_directory) ~= 'string')) then return nil, invalid end
    return { kind = document.kind, preset = document.preset, image_data = document.data,
      classes = document.classes, training = training, supervision = supervision, source = source, runtime = runtime, base_revision = vim.fn.sha256(bytes) }
  end
  if not ok or not bytes or #bytes > 128 * 1024
      or not fields(document, { version = true, inputs = true, outputs = true, routes = true, components = true, component_params = true, encoder_layers = true, encoder_widths = true, native_encoders = true, data = true, training = true, supervision = true })
      or document.version ~= 1 or not vim.islist(document.inputs) or not vim.islist(document.outputs)
      or #document.inputs > 32 or #document.outputs > 32 then return nil, invalid end
  local draft = { inputs = {}, outputs = {}, components = {}, component_params = {}, base_revision = vim.fn.sha256(bytes) }
  if document.supervision ~= nil and document.supervision ~= vim.NIL then
    local ref = document.supervision
    if not fields(ref, { file = true, sha256 = true }) or type(ref.file) ~= 'string' or #ref.file == 0
        or #ref.file > 1024 or ref.file:find('[%c\\:]') or ref.file:sub(1, 1) == '/'
        or ref.file:match('^%./') or ref.file:match('/%./') or ref.file:match('/%.$')
        or ref.file:match('^%.%.$') or ref.file:match('^%.%./') or ref.file:match('/%.%./') or ref.file:match('/%.%.$')
        or type(ref.sha256) ~= 'string' or #ref.sha256 ~= 64 or ref.sha256:find('[^0-9a-f]') then return nil, invalid end
    draft.supervision = { file = ref.file, sha256 = ref.sha256 }
  end
  if document.data ~= nil then
    local data = document.data
    if not fields(data, { factory = true, binding = true, params = true })
        or type(data.factory) ~= 'string' or #data.factory > 240
        or not data.factory:match('^[^%s:;=]+:[^%s:;=]+$')
        or (data.binding ~= nil and not vim.tbl_contains({ 'single', 'single-raw', 'args', 'kwargs', 'envelope' }, data.binding))
        or (data.params ~= nil and (type(data.params) ~= 'table' or (next(data.params) and vim.islist(data.params)))) then return nil, invalid end
    draft.data = { factory = data.factory, binding = data.binding, params = data.params or {} }
  end
  if document.training ~= nil then
    local training = document.training
    if not fields(training, { mode = true, entry = true, params = true, optimizer = true })
        or not vim.tbl_contains({ 'managed-step', 'manual-step', 'external-loop' }, training.mode)
        or type(training.entry) ~= 'string' or #training.entry > 240 or not training.entry:match('^[^%s:;=@]+:[^%s:;=@]+$')
        or (training.params ~= nil and (type(training.params) ~= 'table' or (next(training.params) and vim.islist(training.params)))) then return nil, invalid end
    local optimizer = training.optimizer
    if optimizer == vim.NIL then optimizer = nil end
    if optimizer ~= nil then
      optimizer = vim.deepcopy(optimizer)
      if type(optimizer) == 'table' then
        for _, field in ipairs({ 'factory', 'lr', 'weight_decay' }) do
          if optimizer[field] == vim.NIL then optimizer[field] = nil end
        end
      end
      if not fields(optimizer, { factory = true, params = true, lr = true, weight_decay = true })
          or training.mode == 'external-loop'
          or (optimizer.factory ~= nil and (training.mode ~= 'managed-step' or type(optimizer.factory) ~= 'string'
            or #optimizer.factory > 240 or not optimizer.factory:match('^[^%s:;=@]+:[^%s:;=@]+$')
            or optimizer.lr ~= nil or optimizer.weight_decay ~= nil))
          or (optimizer.params ~= nil and (type(optimizer.params) ~= 'table'
            or (next(optimizer.params) and (not optimizer.factory or vim.islist(optimizer.params)))))
          or (optimizer.lr ~= nil and (type(optimizer.lr) ~= 'number' or optimizer.lr <= 0 or optimizer.lr >= math.huge))
          or (optimizer.weight_decay ~= nil and (type(optimizer.weight_decay) ~= 'number' or optimizer.weight_decay < 0 or optimizer.weight_decay >= math.huge)) then return nil, invalid end
    end
    draft.training = { mode = training.mode, entry = training.entry, params = training.params or {}, optimizer = optimizer }
  end
  local inputs_by_name = {}
  for _, input in ipairs(document.inputs) do
    if not fields(input, { name = true, shape = true, dynamic_axes = true, dtype = true })
        or type(input.name) ~= 'string' or not valid_name(input.name) or inputs_by_name[input.name]
        or not vim.islist(input.shape) or not vim.islist(input.dynamic_axes) then return nil, invalid end
    local axes = {}
    for _, axis in ipairs(input.dynamic_axes) do
      if type(axis) ~= 'number' or axis % 1 ~= 0 or axis < 0 or axis >= #input.shape or axes[axis] then return nil, invalid end
      axes[axis] = true
    end
    local shape = {}
    for index, dim in ipairs(input.shape) do
      if type(dim) ~= 'number' or dim % 1 ~= 0 or dim < 1 or dim > 1000000 then return nil, invalid end
      shape[#shape + 1] = (axes[index - 1] and '?' or '') .. tostring(dim)
    end
    local node = new_input(input.name, table.concat(shape, ','))
    if not parse_shape(node.shape) then return nil, invalid end
    if input.dtype ~= nil and input.dtype ~= vim.NIL then
      if type(input.dtype) ~= 'string' or #input.dtype > 64 or not input.dtype:match('^%a[%w_]*$') then return nil, invalid end
      node.dtype = input.dtype
    end
    draft.inputs[#draft.inputs + 1], inputs_by_name[node.name] = node, node
  end
  for _, output in ipairs(document.outputs) do
    if not fields(output, { name = true, features = true, loss = true, weight = true })
        or type(output.name) ~= 'string' or not valid_name(output.name)
        or type(output.features) ~= 'number' or output.features < 1 or output.features > 65536 or output.features % 1 ~= 0
        or (output.loss ~= 'mse' and output.loss ~= 'cross_entropy'
          and (type(output.loss) ~= 'string' or #output.loss > 247
            or not output.loss:match('^custom:[^%s:;=@]+:[^%s:;=@]+$')))
        or type(output.weight) ~= 'number' or output.weight <= 0 or output.weight >= math.huge then return nil, invalid end
    draft.outputs[#draft.outputs + 1] = {
      name = output.name, width = tostring(output.features), weight = tostring(output.weight),
      loss = output.loss == 'cross_entropy' and 'ce' or output.loss,
    }
  end
  for _, name in ipairs({ 'routes', 'components' }) do
    if document[name] ~= nil and type(document[name]) ~= 'table' then return nil, invalid end
    for slot, value in pairs(document[name] or {}) do
      local input = type(slot) == 'string' and inputs_by_name[(slot:gsub('_encoder$', ''))]
      if input and slot ~= input.name .. '_encoder' then input = nil end
      if type(value) ~= 'string' then return nil, invalid end
      if name == 'routes' then
        if not input or not inputs_by_name[value] then return nil, invalid end
        if inputs_by_name[value] ~= input then input.source = inputs_by_name[value] end
      else
        if not value:match('^[^%s:;=]+:[^%s:;=]+$') or #value > 240 then return nil, invalid end
        if input then input.factory = value
        elseif slot == 'fusion' or slot == 'head' then draft.components[slot] = value
        else return nil, invalid end
      end
    end
  end
  if document.component_params ~= nil and type(document.component_params) ~= 'table' then return nil, invalid end
  for slot, params in pairs(document.component_params or {}) do
    if type(slot) ~= 'string' or not document.components or not document.components[slot]
        or type(params) ~= 'table' or (next(params) and vim.islist(params)) then return nil, invalid end
    for name in pairs(params) do
      if type(name) ~= 'string' or #name > 128 or name == '' or name:find('[%c%s%.=]') then return nil, invalid end
    end
    draft.component_params[slot] = params
  end
  if document.encoder_layers ~= nil and type(document.encoder_layers) ~= 'table' then return nil, invalid end
  local layer_count = 0
  for slot, layers in pairs(document.encoder_layers or {}) do
    local input = type(slot) == 'string' and inputs_by_name[slot:gsub('_encoder$', '')]
    if not input or slot ~= input.name .. '_encoder' or input.factory or not vim.islist(layers)
        or #layers < 1 or #layers > 16 then return nil, invalid end
    for _, layer in ipairs(layers) do if not supported_layer(layer) then return nil, invalid end end
    layer_count = layer_count + count_layers(layers)
    if layer_count > 64 then return nil, invalid end
    input.layers = layers
  end
  if not unique_layer_ids(draft.inputs) then return nil, 'Duplicate layer IDs in model-maven.json.' end
  if document.encoder_widths ~= nil and type(document.encoder_widths) ~= 'table' then return nil, invalid end
  for slot, width in pairs(document.encoder_widths or {}) do
    local input = type(slot) == 'string' and inputs_by_name[slot:gsub('_encoder$', '')]
    if not input or slot ~= input.name .. '_encoder' or not input.factory or input.layers
        or type(width) ~= 'number' or width % 1 ~= 0 or width < 1 or width > 1000000 then return nil, invalid end
    input.component_width = width
  end
  if document.native_encoders ~= nil and not vim.islist(document.native_encoders) then return nil, invalid end
  local seen_native = {}
  for _, slot in ipairs(document.native_encoders or {}) do
    local input = type(slot) == 'string' and inputs_by_name[slot:gsub('_encoder$', '')]
    if not input or slot ~= input.name .. '_encoder' or seen_native[slot] or input.layers then return nil, invalid end
    seen_native[slot], input.native_output = true, true
  end
  return draft
end

-- Recovery snapshots may be incomplete drafts; they are never treated as a
-- saved recipe or as evidence that the generated model has been updated.
function M.restore_snapshot(raw)
  local ok, value = pcall(vim.json.decode, raw)
  local function object(candidate) return type(candidate) == 'table' and (not next(candidate) or not vim.islist(candidate)) end
  local function reference(candidate)
    return type(candidate) == 'string' and #candidate <= 240 and candidate:match('^[^%s:;=]+:[^%s:;=]+$') ~= nil
  end
  if not ok or type(value) ~= 'table' or value.version ~= 1
      or type(value.inputs) ~= 'table' or not vim.islist(value.inputs) or #value.inputs > 32
      or type(value.outputs) ~= 'table' or not vim.islist(value.outputs) or #value.outputs > 32
      or not object(value.components) or not object(value.component_params)
      or (value.base_revision ~= nil and (type(value.base_revision) ~= 'string'
        or #value.base_revision ~= 64 or value.base_revision:find('[^0-9a-f]')))
      or (value.reconcile_assembly ~= nil and type(value.reconcile_assembly) ~= 'boolean') then return nil, 'Invalid recovery draft.' end
  local allowed = {version = true, inputs = true, outputs = true, components = true, component_params = true,
    data = true, training = true, supervision = true, base_revision = true, reconcile_assembly = true}
  for key in pairs(value) do if not allowed[key] then return nil, 'Unsupported recovery draft field.' end end
  local draft = {inputs = {}, outputs = {}, components = value.components, component_params = value.component_params,
    data = value.data, training = value.training, supervision = value.supervision,
    base_revision = value.base_revision, reconcile_assembly = value.reconcile_assembly}
  local source_indices, count = {}, 0
  for index, input in ipairs(value.inputs) do
    if type(input) ~= 'table' or type(input.name) ~= 'string' or not valid_name(input.name)
        or type(input.shape) ~= 'string' or not parse_shape(input.shape) or type(input.dtype) ~= 'string'
        or (input.dtype ~= '' and (#input.dtype > 64 or not input.dtype:match('^%a[%w_]*$')))
        or (input.factory ~= nil and not reference(input.factory))
        or (input.component_width ~= nil and (type(input.component_width) ~= 'number' or input.component_width % 1 ~= 0 or input.component_width < 1 or input.component_width > 1000000))
        or (input.native_output ~= nil and type(input.native_output) ~= 'boolean')
        or (input.layers ~= nil and (type(input.layers) ~= 'table' or not vim.islist(input.layers) or #input.layers > 16)) then return nil, 'Invalid recovery input.' end
    for key in pairs(input) do if not ({name=true, shape=true, dtype=true, source_index=true, factory=true,
      component_width=true, native_output=true, layers=true})[key] then return nil, 'Unsupported recovery input field.' end end
    for _, layer in ipairs(input.layers or {}) do if not supported_layer(layer) then return nil, 'Invalid recovery layer.' end end
    count = count + count_layers(input.layers)
    if count > 64 then return nil, 'Recovery layer limit exceeded.' end
    source_indices[index] = input.source_index
    input.source_index = nil
    draft.inputs[index] = input
  end
  for index, source in pairs(source_indices) do
    if type(source) ~= 'number' or source % 1 ~= 0 or not draft.inputs[source] then return nil, 'Invalid recovery source.' end
    draft.inputs[index].source = draft.inputs[source]
  end
  if not unique_layer_ids(draft.inputs) then return nil, 'Duplicate layer IDs in recovery draft.' end
  for _, output in ipairs(value.outputs) do
    if type(output) ~= 'table' or type(output.name) ~= 'string' or not valid_name(output.name)
        or type(output.width) ~= 'string' or not tonumber(output.width) or tonumber(output.width) < 1
        or tonumber(output.width) > 65536 or tonumber(output.width) % 1 ~= 0
        or type(output.weight) ~= 'string' or not tonumber(output.weight) or tonumber(output.weight) <= 0
        or tonumber(output.weight) >= math.huge or type(output.loss) ~= 'string'
        or not (output.loss == 'mse' or output.loss == 'ce' or output.loss:match('^custom:[^%s:;=@]+:[^%s:;=@]+$')) then return nil, 'Invalid recovery output.' end
    for key in pairs(output) do if not ({name=true, width=true, weight=true, loss=true})[key] then return nil, 'Unsupported recovery output field.' end end
    draft.outputs[#draft.outputs + 1] = output
  end
  for slot, factory in pairs(draft.components) do
    if type(slot) ~= 'string' or not reference(factory) then return nil, 'Invalid recovery component.' end
  end
  for slot, params in pairs(draft.component_params) do
    if type(slot) ~= 'string' or not object(params) then return nil, 'Invalid recovery component parameters.' end
  end
  if draft.data ~= nil and (type(draft.data) ~= 'table' or not reference(draft.data.factory) or not object(draft.data.params)
      or (draft.data.binding ~= nil and not vim.tbl_contains({'single', 'single-raw', 'args', 'kwargs', 'envelope'}, draft.data.binding))) then return nil, 'Invalid recovery data.' end
  if draft.training ~= nil and (type(draft.training) ~= 'table' or type(draft.training.entry) ~= 'string'
      or not vim.tbl_contains({'managed-step', 'manual-step', 'external-loop'}, draft.training.mode)
      or not (draft.training.entry == '$model:step' or reference(draft.training.entry))
      or (draft.training.mode == 'external-loop' and draft.training.entry == '$model:step')
      or not object(draft.training.params) or (draft.training.optimizer ~= nil and not object(draft.training.optimizer))) then return nil, 'Invalid recovery training.' end
  local optimizer = draft.training and draft.training.optimizer
  if optimizer and ((optimizer.factory ~= nil and not reference(optimizer.factory))
      or (optimizer.params ~= nil and not object(optimizer.params))
      or (optimizer.lr ~= nil and (type(optimizer.lr) ~= 'number' or optimizer.lr <= 0 or optimizer.lr >= math.huge))
      or (optimizer.weight_decay ~= nil and (type(optimizer.weight_decay) ~= 'number' or optimizer.weight_decay < 0 or optimizer.weight_decay >= math.huge))) then
    return nil, 'Invalid recovery optimizer.'
  end
  if draft.supervision ~= nil and (type(draft.supervision) ~= 'table' or type(draft.supervision.file) ~= 'string'
      or type(draft.supervision.sha256) ~= 'string') then return nil, 'Invalid recovery supervision.' end
  return draft
end

function M.sync_assembly(draft, bindings, routes, modes)
  if type(bindings) ~= 'table' then return nil, 'Invalid saved component bindings.' end
  if routes ~= nil and (type(routes) ~= 'table' or (next(routes) and vim.islist(routes))) then
    return nil, 'Invalid saved input routes.'
  end
  if modes ~= nil and (type(modes) ~= 'table' or (next(modes) and vim.islist(modes))) then
    return nil, 'Invalid saved encoder output modes.'
  end
  local by_slot, by_name = { fusion = true, head = true }, {}
  for _, input in ipairs(draft.inputs) do by_slot[input.name .. '_encoder'], by_name[input.name] = input, input end
  local resolved_routes = {}
  for slot, source in pairs(routes or {}) do
    if type(slot) ~= 'string' or not by_slot[slot] or slot == 'fusion' or slot == 'head'
        or type(source) ~= 'string' or not by_name[source] then
      return nil, 'A saved route has no corresponding canvas node: ' .. tostring(slot)
    end
    resolved_routes[slot] = by_name[source]
  end
  local resolved_modes = {}
  for slot, mode in pairs(modes or {}) do
    local node = by_slot[slot]
    if type(slot) ~= 'string' or type(node) ~= 'table' or (mode ~= 'native' and mode ~= 'flatten') then
      return nil, 'A saved encoder output mode cannot be represented: ' .. tostring(slot)
    end
    resolved_modes[slot] = mode
  end
  local resolved, resolved_params = {}, {}
  for slot, value in pairs(bindings) do
    if not by_slot[slot] then return nil, 'A saved component has no corresponding canvas node: ' .. tostring(slot) end
    local reference = value
    if type(value) == 'table' then
      local component = value.component_ref
      if type(component) == 'table' and type(component.id) == 'string' and type(component.version) == 'string' then
        reference = 'component:' .. component.id .. '@' .. component.version
      else
        reference = value.factory
      end
    end
    if type(reference) ~= 'string' or #reference > 240 or not reference:match('^[^%s:;=]+:[^%s:;=]+$') then
      return nil, 'A saved component cannot be represented in the canvas: ' .. tostring(slot)
    end
    resolved[slot] = reference
    if type(value) == 'table' and type(value.params) == 'table' and next(value.params) then
      resolved_params[slot] = vim.deepcopy(value.params)
    end
  end
  for slot, mode in pairs(resolved_modes) do
    local node = by_slot[slot]
    if mode == 'native' and node.layers and #node.layers > 0 and not resolved[slot] then
      return nil, 'Native output is not supported for a built-in layer stack: ' .. slot
    end
  end
  draft.components = {}
  draft.component_params = resolved_params
  for _, input in ipairs(draft.inputs) do
    local source = resolved_routes[input.name .. '_encoder']
    if source then input.source = source ~= input and source or nil end
    local mode = resolved_modes[input.name .. '_encoder']
    if mode then input.native_output = mode == 'native' end
    if resolved[input.name .. '_encoder'] and input.layers and #input.layers > 0 then
      local width
      for _, layer in ipairs(input.layers) do if layer.features then width = layer.features end end
      input.component_width = width
    end
    input.factory = resolved[input.name .. '_encoder']
    if input.factory then input.layers = nil else input.component_width = nil end
  end
  for _, slot in ipairs({ 'fusion', 'head' }) do draft.components[slot] = resolved[slot] end
  draft.reconcile_assembly = true
  return true
end

function M.assembly_matches(draft, bindings, routes, modes)
  local copy = vim.deepcopy(draft)
  local ok, err = M.sync_assembly(copy, bindings, routes, modes)
  if not ok then return nil, err end
  local function signature(value)
    local result = { inputs = {}, components = value.components or {}, params = {} }
    for _, input in ipairs(value.inputs) do
      result.inputs[#result.inputs + 1] = { name = input.name, source = (input.source or input).name,
        factory = input.factory, native = input.native_output == true }
    end
    for slot, params in pairs(value.component_params or {}) do
      if type(params) == 'table' and next(params) then result.params[slot] = params end
    end
    return result
  end
  return vim.deep_equal(signature(draft), signature(copy))
end

-- Compare only persisted input settings; a session override is deliberately
-- not treated as a project or declaration change.
function M.data_matches(draft, config)
  if type(draft) ~= 'table' or type(config) ~= 'table' or type(config.batch) ~= 'string' then return nil end
  local params = config.batch_params
  if params == nil or params == vim.NIL then params = {} end
  if type(params) ~= 'table' or (next(params) and vim.islist(params)) then return nil end
  local binding = config.batch_binding
  if binding == vim.NIL then binding = nil end
  if binding ~= nil and type(binding) ~= 'string' then return nil end
  local data = draft.data
  if data == nil then
    local module = type(config.model) == 'string' and config.model:match('^([^:]+):model$') or nil
    if not module or not (module == 'model_starter' or module:match('^model_maven_build_[a-f0-9]+$')) then return nil end
    return config.batch == module .. ':batch' and binding == 'envelope' and not next(params)
  end
  if type(data) ~= 'table' or type(data.factory) ~= 'string' or type(data.params or {}) ~= 'table' then return nil end
  local expected_binding = data.binding ~= vim.NIL and data.binding or nil
  return config.batch == data.factory and binding == expected_binding and vim.deep_equal(params, data.params or {})
end

local function fit(text, width)
  text = text:gsub('[%c]', ' ')
  if vim.fn.strdisplaywidth(text) <= width then return text end
  local n = vim.fn.strchars(text)
  repeat
    n = n - 1
    text = vim.fn.strcharpart(text, 0, n)
  until n == 0 or vim.fn.strdisplaywidth(text) < width
  return text .. '…'
end

-- A row can contain independent clickable cells. All ranges use byte offsets,
-- matching the columns Neovim reports for mouse events and highlights.
local function cell(text, width, style, activate)
  text = fit(text, width)
  return { text = text .. string.rep(' ', width - vim.fn.strdisplaywidth(text)), style = style, activate = activate }
end

function M.open(draft, actions)
  local width = math.min(116, vim.o.columns - 4)
  local height = math.min(35, vim.o.lines - 5)
  if width < 28 or height < 8 then
    vim.notify(label('Enlarge the terminal to open the builder.', '请放大终端以打开构建器。', '請放大終端以開啟建構器。'), vim.log.levels.WARN)
    return
  end
  local buf = api.nvim_create_buf(false, true)
  vim.bo[buf].bufhidden = 'wipe'
  local win = api.nvim_open_win(buf, true, {
    relative = 'editor', style = 'minimal', border = 'rounded',
    width = width, height = height, row = 1, col = math.floor((vim.o.columns - width) / 2),
  })
  vim.wo[win].wrap = false
  local ns = api.nvim_create_namespace('ModelMavenBuilder')
  local targets, component_choices = {}, nil
  local kind, selected, tab, message = 'fusion', nil, 'properties', nil
  local picker, layer_index, layer_catalog, factory_layers, residual_layers, parallel_layers, custom_losses, layer_plan_capability, layer_edit_capability, layer_cross_copy_capability, drag
  local id_capability, ids_enabled = false, false
  local branch_selected, route_pending
  local layer_query = ''
  -- Observations are session evidence, not part of the model declaration.
  local layer_signatures = setmetatable({}, { __mode = 'k' })
  local render, recipe, plan_layer_move, plan_layer_list_change, plan_layer_value_change, plan_layer_copy, plan_layer_structure
  local move_undo, move_redo = {}, {}
  draft.components = draft.components or {}
  draft.component_params = draft.component_params or {}
  local function id_state()
    local used, missing, has_any = {}, {}, false
    for _, input in ipairs(draft.inputs or {}) do
      visit_layers(input.layers, function(layer)
        if layer.id then used[layer.id], has_any = true, true else missing[#missing + 1] = layer end
      end)
    end
    return used, missing, has_any
  end
  local function assign_layer_ids()
    if not unique_layer_ids(draft.inputs) then return nil, 'Duplicate layer IDs.' end
    local used, missing = id_state()
    local assigned = {}
    for _, layer in ipairs(missing) do
      local id
      for _ = 1, 16 do
        local bytes, error = vim.uv.random(16)
        if not bytes then return nil, error end
        local candidate = 'n_' .. bytes:gsub('.', function(char) return string.format('%02x', string.byte(char)) end)
        if not used[candidate] then id = candidate; used[id] = true; break end
      end
      if not id then return nil, 'Could not allocate a unique layer ID.' end
      assigned[#assigned + 1] = {layer, id}
    end
    for _, entry in ipairs(assigned) do entry[1].id = entry[2] end
    return true
  end
  local function assign_detached_ids(layer, reserved)
    if not ids_enabled then return true end
    local used, assigned, failure = id_state(), {}, nil
    for id in pairs(reserved or {}) do used[id] = true end
    visit_layers({layer}, function(item)
      if failure then return end
      if item.id then
        if used[item.id] then failure = 'Duplicate layer ID.' else used[item.id] = true; if reserved then reserved[item.id] = true end end
        return
      end
      local id
      for _ = 1, 16 do
        local bytes, error = vim.uv.random(16)
        if not bytes then failure = error; return end
        local candidate = 'n_' .. bytes:gsub('.', function(char) return string.format('%02x', string.byte(char)) end)
        if not used[candidate] then id = candidate; used[id] = true; if reserved then reserved[id] = true end; break end
      end
      if not id then failure = 'Could not allocate a unique layer ID.'; return end
      assigned[#assigned + 1] = {item, id}
    end)
    if failure then return nil, failure end
    for _, item in ipairs(assigned) do item[1].id = item[2] end
    return true
  end
  local function alive() return api.nvim_win_is_valid(win) and api.nvim_win_get_buf(win) == buf end
  local function close() if alive() then api.nvim_win_close(win, true) end end
  local function select_node(next_kind, row)
    branch_selected = nil
    kind, selected, tab, picker, message, layer_index = next_kind, row, 'properties', nil, nil, nil
    render('inspector')
  end
  local function same_order(layers, snapshot)
    if type(layers) ~= 'table' or #layers ~= #snapshot then return false end
    for i, layer in ipairs(snapshot) do if layers[i] ~= layer then return false end end
    return true
  end
  local function copy_order(layers)
    local snapshot = {}
    for i, layer in ipairs(layers) do snapshot[i] = layer end
    return snapshot
  end
  local function find_branch_layer(parent, target)
    if not parent then return end
    for _, group in ipairs(branch_groups(parent)) do
      for index, child in ipairs(group) do
        if child == target then return group, index end
        local found, child_index = find_branch_layer(child, target)
        if found then return found, child_index end
      end
    end
  end
  local function record_layer_list(owner, before, layer)
    move_undo[#move_undo + 1] = { kind = 'list', owner = owner, layer = layer, before = before, after = copy_order(owner.layers),
      before_values = vim.deepcopy(before), after_values = vim.deepcopy(owner.layers) }
    if #move_undo > 64 then table.remove(move_undo, 1) end
    move_redo = {}
  end
  local function record_layer_batch(owner, layer, changes)
    move_undo[#move_undo + 1] = {kind = 'batch', owner = owner, layer = layer, changes = changes}
    if #move_undo > 64 then table.remove(move_undo, 1) end
    move_redo = {}
  end
  -- History retains layer objects by stable ID so undoing a nested edit does not
  -- orphan objects held by an earlier wrap, unwrap, move, or copy record.
  local function capture_layer_refs(root, refs)
    visit_layers({root}, function(item)
      if item.id then refs[item.id] = item end
    end)
  end
  local function restore_layer(target, snapshot, refs)
    local restored = {}
    for key, value in pairs(snapshot) do
      if key ~= 'branch' and key ~= 'branches' then restored[key] = vim.deepcopy(value) end
    end
    local function restore_group(group)
      local result = {}
      for index, item in ipairs(group) do
        local child = item.id and refs[item.id] or nil
        child = child or {}
        restore_layer(child, item, refs)
        result[index] = child
      end
      return result
    end
    if snapshot.branch then restored.branch = restore_group(snapshot.branch) end
    if snapshot.branches then
      restored.branches = {}
      for index, group in ipairs(snapshot.branches) do restored.branches[index] = restore_group(group) end
    end
    for key in pairs(target) do target[key] = nil end
    for key, value in pairs(restored) do target[key] = value end
  end
  local function connect_input(owner, source)
    if not owner or not source or not vim.tbl_contains(draft.inputs, owner) or not vim.tbl_contains(draft.inputs, source) then return false end
    local before = owner.source or owner
    if before == source then return true end
    owner.source = source ~= owner and source or nil
    move_undo[#move_undo + 1] = {kind = 'route', owner = owner, before = before, after = source}
    if #move_undo > 64 then table.remove(move_undo, 1) end
    move_redo = {}
    message = label('Input connected in draft; review dimensions and generate to save.',
      '已在草稿中接线；请审核维度并生成保存。', '已在草稿中接線；請檢視維度並產生儲存。')
    return true
  end
  local function keyboard_connect()
    if kind == 'input' and selected and vim.tbl_contains(draft.inputs, selected) then
      route_pending = {source = selected, document = recipe()}
      message = label('Choose an encoder and press c to connect; Esc cancels.',
        '选择编码器后按 c 接线；Esc 取消。', '選擇編碼器後按 c 接線；Esc 取消。')
    elseif kind == 'encoder' and selected and route_pending then
      local pending = route_pending
      route_pending = nil
      if not vim.tbl_contains(draft.inputs, pending.source) or not vim.tbl_contains(draft.inputs, selected)
          or not vim.deep_equal(recipe(), pending.document) then
        message = label('Draft changed; connection cancelled.', '草稿已变化，接线已取消。', '草稿已變更，接線已取消。')
      else connect_input(selected, pending.source) end
    else
      message = label('Select an input and press c to start a connection.', '请选择输入并按 c 开始接线。', '請選擇輸入並按 c 開始接線。')
    end
    render('inspector')
  end
  local function edit_layer(owner, layer, apply)
    if not owner or not vim.tbl_contains(draft.inputs, owner) or type(owner.layers) ~= 'table'
        or not vim.tbl_contains(owner.layers, layer) then return end
    local before, refs = vim.deepcopy(layer), {}
    capture_layer_refs(layer, refs)
    apply()
    if ids_enabled then
      local ok, error = assign_layer_ids()
      if not ok then ids_enabled = false; message = label('Layer identity could not be assigned: ', '无法分配层身份：', '無法指派層身分：') .. tostring(error) end
    end
    if not vim.deep_equal(before, layer) then
      move_undo[#move_undo + 1] = { kind = 'value', owner = owner, layer = layer,
        before = before, after = vim.deepcopy(layer), refs = refs }
      capture_layer_refs(layer, refs)
      if #move_undo > 64 then table.remove(move_undo, 1) end
      move_redo = {}
    end
  end
  local function move_layer(owner, from, destination)
    local layers = owner and owner.layers
    if type(layers) ~= 'table' or not layers[from] or destination < 1 or destination > #layers then return false end
    if from == destination then return true end
    plan_layer_move(owner, layers, from, destination, nil)
    return true
  end
  local function undo_move(redo)
    local from, to = redo and move_redo or move_undo, redo and move_undo or move_redo
    local record = from[#from]
    if not record then message = label('No layer change to undo or redo.', '没有可撤销或重做的层修改。', '沒有可復原或重做的層修改。'); render(); return end
    local expected, result = redo and record.before or record.after, redo and record.after or record.before
    local matching = record.kind == 'value'
      and type(record.owner.layers) == 'table' and vim.tbl_contains(record.owner.layers, record.layer)
      and vim.deep_equal(record.layer, expected)
      or record.kind == 'list' and same_order(record.owner.layers, expected)
        and vim.deep_equal(record.owner.layers, redo and record.before_values or record.after_values)
        and vim.deep_equal(result, redo and record.after_values or record.before_values)
    if record.kind == 'route' then
      matching = (record.owner.source or record.owner) == expected
        and vim.tbl_contains(draft.inputs, expected) and vim.tbl_contains(draft.inputs, result)
    end
    if record.kind == 'batch' then
      matching = type(record.owner.layers) == 'table' and vim.tbl_contains(record.owner.layers, record.layer)
      for _, change in ipairs(record.changes) do
        local present = true
        if redo then present = change.before_present end
        local expected_order = redo and change.before or change.after
        if not vim.tbl_contains(draft.inputs, change.owner)
            or (present and not same_order(change.owner.layers, expected_order))
            or (not present and change.owner.layers ~= nil)
            or not vim.deep_equal(change.copied, change.copied_value) then matching = false; break end
      end
    end
    if not vim.tbl_contains(draft.inputs, record.owner) or not matching then
      message = label('The encoder or layer list changed; this action cannot be safely restored.',
        '编码器或层列表已变化，无法安全恢复此操作。', '編碼器或層列表已變更，無法安全還原此操作。'); render(); return
    end
    table.remove(from)
    branch_selected = nil
    if record.kind == 'batch' then
      for _, change in ipairs(record.changes) do
        if redo then change.owner.layers = copy_order(change.after)
        elseif change.before_present then change.owner.layers = copy_order(change.before)
        else change.owner.layers = nil end
      end
    elseif record.kind == 'route' then
      record.owner.source = result ~= record.owner and result or nil
    elseif record.kind == 'value' then
      restore_layer(record.layer, result, record.refs)
      layer_signatures[record.layer] = nil
    else record.owner.layers = copy_order(result) end
    to[#to + 1] = record
    kind, selected, layer_index = 'encoder', record.owner, nil
    for index, layer in ipairs(record.owner.layers or {}) do if layer == record.layer then layer_index = index; break end end
    message = redo and label('Change redone.', '已重做修改。', '已重做修改。')
      or label('Change undone.', '已撤销修改。', '已復原修改。')
    render()
  end
  local function edit(title, initial, validate, apply)
    vim.ui.input({ prompt = title, default = initial }, function(raw)
      if raw == nil or not alive() then return end
      local value = validate(vim.trim(raw))
      if value ~= nil then message = nil; apply(value)
      else message = label('Invalid value. Check the field.', '值无效，请检查该字段。', '值無效，請檢查該欄位。') end
      render('inspector')
    end)
  end
  local function edit_parameters(current, apply, asynchronous)
    local values = vim.deepcopy(current or vim.empty_dict())
    local choose
    local function commit()
      local ok, encoded = pcall(vim.json.encode, next(values) and values or vim.empty_dict())
      if not ok or #encoded > 65536 or #vim.tbl_keys(values) > 64 then
        message = label('Parameters are too large.', '参数过大。', '參數過大。'); render('inspector'); return false
      end
      local updated = next(values) and vim.deepcopy(values) or vim.empty_dict()
      if asynchronous then
        apply(updated, function(success) if success and alive() then choose() end end)
      else
        apply(updated)
        message = nil; render('inspector'); choose()
      end
      return true
    end
    local function enter_value(name)
      local default = values[name] ~= nil and vim.json.encode(values[name]) or ''
      vim.ui.input({ prompt = label('Value (JSON or plain text): ', '值（JSON 或普通文本）：', '值（JSON 或一般文字）：') .. name .. ' = ', default = default }, function(raw)
        if raw == nil or not alive() then return end
        local ok, value = pcall(vim.json.decode, raw)
        if not ok then value = raw end
        values[name] = value
        commit()
      end)
    end
    choose = function()
      if not alive() then return end
      local names = vim.tbl_keys(values); table.sort(names)
      local choices = { label('Add parameter', '添加参数', '新增參數'), label('Done', '完成', '完成') }
      for _, name in ipairs(names) do choices[#choices + 1] = name .. ' = ' .. vim.json.encode(values[name]) end
      vim.ui.select(choices, { prompt = label('Factory parameters', '工厂参数', '工廠參數') }, function(_, index)
        if not index or not alive() or index == 2 then return end
        if index == 1 then
          vim.ui.input({ prompt = label('Parameter name: ', '参数名：', '參數名稱：') }, function(name)
            if name == nil or not alive() then return end
            name = vim.trim(name)
            if name == '' or #name > 128 or name:find('[%c%s%.=]') or values[name] ~= nil or #names >= 64 then
              message = label('Invalid or duplicate parameter name.', '参数名无效或重复。', '參數名稱無效或重複。'); render('inspector'); return
            end
            enter_value(name)
          end)
          return
        end
        local name = names[index - 2]
        if not name then return end
        vim.ui.select({ label('Edit value', '修改值', '修改值'), label('Remove parameter', '删除参数', '刪除參數'), label('Back', '返回', '返回') },
          { prompt = name }, function(_, choice)
            if not alive() or not choice then return end
            if choice == 1 then enter_value(name)
            elseif choice == 2 then values[name] = nil; commit()
            else choose() end
          end)
      end)
    end
    choose()
  end
  local function add(which)
    local rows = which == 'input' and draft.inputs or draft.outputs
    if #rows >= 32 then
      message = label('Limit: 32 per group.', '每组最多 32 项。', '每組最多 32 項。')
      render('inspector')
      return
    end
    local name = unique_name(rows, which)
    local node = which == 'input' and new_input(name, '4') or new_output(name)
    rows[#rows + 1] = node
    select_node(which, node)
  end
  local function example()
    if #draft.inputs > 0 or #draft.outputs > 0 or next(draft.components) then
      message = label('The example needs a blank draft.', '请在空白草稿中载入示例。', '請在空白草稿中載入範例。')
      render('inspector')
      return
    end
    draft.inputs = { new_input('features', '4'), new_input('context', '?8,4') }
    draft.outputs = { new_output('score') }
    select_node('encoder', draft.inputs[1])
  end
  local function classification()
    local input, output = new_input('features', '4'), new_output('class_scores')
    input.layers = { { type = 'linear', features = 16 }, { type = 'relu' } }
    output.width, output.loss = '2', 'ce'
    draft.inputs, draft.outputs = { input }, { output }
    select_node('encoder', input)
  end
  local function sequence()
    local input = new_input('input1', '4')
    draft.inputs, draft.outputs = { input }, { new_output('output1') }
    select_node('encoder', input)
  end
  recipe = function()
    local config = { version = 1, inputs = {}, outputs = {}, routes = {}, encoder_layers = {}, encoder_widths = {}, native_encoders = {}, components = vim.deepcopy(draft.components), component_params = vim.deepcopy(draft.component_params) }
    if draft.training then
      config.training = vim.deepcopy(draft.training)
      if not next(config.training.params) then config.training.params = nil end
      if config.training.optimizer and config.training.optimizer.params and not next(config.training.optimizer.params) then
        config.training.optimizer.params = nil
      end
    end
    if draft.supervision then config.supervision = vim.deepcopy(draft.supervision) end
    if draft.data then
      config.data = vim.deepcopy(draft.data)
      if not next(config.data.params) then config.data.params = nil end
    end
    for _, input in ipairs(draft.inputs) do
      local shape, axes = parse_shape(input.shape)
      config.inputs[#config.inputs + 1] = {
        name = input.name, shape = shape, dynamic_axes = axes,
        dtype = input.dtype ~= '' and input.dtype or vim.NIL,
      }
      local slot = input.name .. '_encoder'
      if input.native_output then config.native_encoders[#config.native_encoders + 1] = slot end
      if input.source and input.source ~= input then config.routes[slot] = input.source.name end
      if input.factory then config.components[slot] = input.factory end
      if input.layers and #input.layers > 0 then config.encoder_layers[slot] = vim.deepcopy(input.layers) end
      if input.factory and input.component_width then config.encoder_widths[slot] = input.component_width end
    end
    for _, output in ipairs(draft.outputs) do
      config.outputs[#config.outputs + 1] = { name = output.name, features = tonumber(output.width),
        loss = output.loss == 'ce' and 'cross_entropy' or output.loss, weight = tonumber(output.weight) }
    end
    if not next(config.routes) then config.routes = nil end
    if not next(config.components) then config.components = nil end
    if not next(config.encoder_layers) then config.encoder_layers = nil end
    if not next(config.encoder_widths) then config.encoder_widths = nil end
    if #config.native_encoders == 0 then config.native_encoders = nil end
    if not next(config.component_params) then config.component_params = nil end
    return config
  end
  plan_layer_move = function(owner, layers, from, destination, root)
    local before, layer = copy_order(layers), layers[from]
    local selected_index, selected_branch = layer_index, branch_selected
    if not layer or destination < 1 or destination > #layers then return end
    local desired = copy_order(layers)
    table.remove(desired, from); table.insert(desired, destination, layer)
    local order = {}
    for _, item in ipairs(desired) do order[#order + 1] = item.id end
    local document = recipe()
    local function apply(plan)
      if not alive() or selected ~= owner or not vim.tbl_contains(draft.inputs, owner)
          or layer_index ~= selected_index or branch_selected ~= selected_branch
          or not same_order(layers, before) or not vim.deep_equal(recipe(), document)
          or (root and find_branch_layer(root, layer) ~= layers) or (not root and owner.layers ~= layers) then
        message = label('The draft changed while planning; move cancelled.', '规划期间草稿已变化，移动已取消。', '規劃期間草稿已變更，移動已取消。')
        if alive() then render('inspector') end
        return
      end
      if plan and not vim.deep_equal(plan.outcomes[1].order, order) then
        message = label('The backend order differs; move cancelled.', '后端顺序不一致，移动已取消。', '後端順序不一致，移動已取消。')
        return render('inspector')
      end
      if root then
        edit_layer(owner, root, function() table.remove(layers, from); table.insert(layers, destination, layer) end)
        branch_selected = layer
      else
        table.remove(layers, from); table.insert(layers, destination, layer)
        record_layer_list(owner, before, layer)
        kind, selected, layer_index = 'encoder', owner, destination
      end
      local validation = plan and plan.validation
      if validation and validation.status == 'unavailable_or_invalid' then
        message = label('Moved in draft; static generation needs attention: ', '已移动草稿；静态生成需处理：', '已移動草稿；靜態產生需處理：') .. tostring(validation.message)
      elseif not plan then
        message = label('Moved locally; backend planning needs a compatible CLI and stable IDs.',
          '已在本地移动；后端规划需要兼容的 CLI 和稳定 ID。', '已在本機移動；後端規劃需要相容的 CLI 和穩定 ID。')
      else message = label('Backend planned the move; review and generate to save.', '后端已规划移动，审核生成后保存。', '後端已規劃移動，檢視產生後儲存。') end
      render('inspector')
    end
    if layer_plan_capability and actions.plan_layer_move and layer.id and #order == #desired then
      local next_layer = desired[destination + 1]
      message = label('Planning layer move…', '正在规划层移动…', '正在規劃層移動…')
      render('inspector')
      actions.plan_layer_move(document, {id = layer.id, before = next_layer and next_layer.id or vim.NIL}, function(plan, error)
        if not alive() then return end
        if not plan then message = error; return render('inspector') end
        apply(plan)
      end)
    else apply(nil) end
  end
  plan_layer_list_change = function(owner, layers, index, action, layer, signature, root)
    local before, selection, current_picker, selected_branch = copy_order(layers), layer_index, picker, branch_selected
    if action == 'add' then
      local ok, error = assign_detached_ids(layer)
      if not ok then message = label('Could not assign layer IDs: ', '无法分配层身份：', '無法指派層身分：') .. tostring(error); return render('inspector') end
    end
    local desired = copy_order(layers)
    if action == 'add' then table.insert(desired, index, layer) else table.remove(desired, index) end
    local order = {}
    for _, item in ipairs(desired) do order[#order + 1] = item.id end
    local document = recipe()
    local operation = action == 'add'
      and {action = 'add', slot = owner.name .. '_encoder', before = layers[index] and layers[index].id or vim.NIL,
        after = index > #layers and #layers > 0 and layers[#layers].id or vim.NIL, layer = layer}
      or {action = 'remove', id = layer.id}
    local function apply(plan)
      if not alive() or selected ~= owner or not vim.tbl_contains(draft.inputs, owner)
          or type(owner.layers) ~= 'table' or (not root and owner.layers ~= layers)
          or (root and (owner.layers[selection] ~= root or not before[1] or find_branch_layer(root, before[1]) ~= layers))
          or layer_index ~= selection or picker ~= current_picker or branch_selected ~= selected_branch
          or not same_order(layers, before) or not vim.deep_equal(recipe(), document)
          or (action == 'remove' and layers[index] ~= layer) then
        message = label('The draft changed while planning; edit cancelled.', '规划期间草稿已变化，编辑已取消。', '規劃期間草稿已變更，編輯已取消。')
        if alive() then render('inspector') end
        return
      end
      if plan and (plan.outcomes[1].id ~= layer.id or plan.outcomes[1].slot ~= owner.name .. '_encoder'
          or not vim.deep_equal(plan.outcomes[1].order, order)) then
        message = label('The backend result differs; edit cancelled.', '后端结果不一致，编辑已取消。', '後端結果不一致，編輯已取消。')
        return render('inspector')
      end
      if root then
        edit_layer(owner, root, function()
          if action == 'add' then table.insert(layers, index, layer) else table.remove(layers, index) end
        end)
        branch_selected = action == 'add' and layer or nil
      else
        if action == 'add' then
          table.insert(layers, index, layer)
          if type(signature) == 'table' then layer_signatures[layer] = signature end
          layer_index, picker = index, nil
        else
          table.remove(layers, index)
          layer_index = #layers > 0 and math.min(index, #layers) or nil
        end
        record_layer_list(owner, before, layer)
      end
      local validation = plan and plan.validation
      if validation and validation.status == 'unavailable_or_invalid' then
        message = label('Draft updated; static generation needs attention: ', '草稿已更新；静态生成需处理：', '草稿已更新；靜態產生需處理：') .. tostring(validation.message)
      elseif plan then message = label('Backend planned the edit; review and generate to save.', '后端已规划编辑，审核生成后保存。', '後端已規劃編輯，檢視產生後儲存。')
      else message = label('Edited locally; backend planning needs a compatible CLI and stable IDs.',
        '已在本地编辑；后端规划需要兼容的 CLI 和稳定 ID。', '已在本機編輯；後端規劃需要相容的 CLI 和穩定 ID。') end
      render('inspector')
    end
    if layer_edit_capability and actions.plan_layer_edit and layer.id and #order == #desired then
      message = label('Planning layer edit…', '正在规划层编辑…', '正在規劃層編輯…')
      render('inspector')
      actions.plan_layer_edit(document, operation, function(plan, error)
        if not alive() then return end
        if not plan then message = error; return render('inspector') end
        apply(plan)
      end)
    else apply(nil) end
  end
  plan_layer_value_change = function(owner, root, target, fields, done)
    local selection, selected_branch, current_picker = layer_index, branch_selected, picker
    local group = target == root and owner.layers or find_branch_layer(root, target)
    if not group or not vim.tbl_contains(group, target) then if done then done(false) end; return end
    local before, document = copy_order(group), recipe()
    local order = {}
    for _, item in ipairs(group) do order[#order + 1] = item.id end
    local operation = {action = 'set', id = target.id, fields = vim.deepcopy(fields)}
    local function apply(plan)
      local current_group = target == root and owner.layers or find_branch_layer(root, target)
      if not alive() or selected ~= owner or not vim.tbl_contains(draft.inputs, owner) or type(owner.layers) ~= 'table'
          or owner.layers[selection] ~= root or layer_index ~= selection or branch_selected ~= selected_branch
          or picker ~= current_picker or current_group ~= group or not same_order(group, before)
          or not vim.deep_equal(recipe(), document) then
        message = label('The draft changed while planning; edit cancelled.', '规划期间草稿已变化，编辑已取消。', '規劃期間草稿已變更，編輯已取消。')
        if alive() then render('inspector') end
        if done then done(false) end
        return
      end
      if plan and (plan.outcomes[1].id ~= target.id or plan.outcomes[1].slot ~= owner.name .. '_encoder'
          or not vim.deep_equal(plan.outcomes[1].order, order)) then
        message = label('The backend result differs; edit cancelled.', '后端结果不一致，编辑已取消。', '後端結果不一致，編輯已取消。')
        render('inspector'); if done then done(false) end; return
      end
      edit_layer(owner, root, function()
        if fields.reference ~= nil and target.reference ~= fields.reference then layer_signatures[target] = nil end
        for key, value in pairs(fields) do target[key] = vim.deepcopy(value) end
      end)
      local validation = plan and plan.validation
      if validation and validation.status == 'unavailable_or_invalid' then
        message = label('Draft updated; static generation needs attention: ', '草稿已更新；静态生成需处理：', '草稿已更新；靜態產生需處理：') .. tostring(validation.message)
      elseif plan then message = label('Backend planned the edit; review and generate to save.', '后端已规划编辑，审核生成后保存。', '後端已規劃編輯，檢視產生後儲存。')
      else message = label('Edited locally; backend planning needs a compatible CLI and stable IDs.',
        '已在本地编辑；后端规划需要兼容的 CLI 和稳定 ID。', '已在本機編輯；後端規劃需要相容的 CLI 和穩定 ID。') end
      render('inspector')
      if done then done(true) end
    end
    if layer_edit_capability and actions.plan_layer_edit and target.id and #order == #group then
      message = label('Planning layer edit…', '正在规划层编辑…', '正在規劃層編輯…')
      render('inspector')
      actions.plan_layer_edit(document, operation, function(plan, error)
        if not alive() then return end
        if not plan then message = error; render('inspector'); if done then done(false) end; return end
        apply(plan)
      end)
    else apply(nil) end
  end
  -- Replace a contiguous top-level range in one backend plan, retaining child identities.
  plan_layer_structure = function(owner, first, last, inserted)
    local layers, selection, selected_branch, current_picker = owner.layers, layer_index, branch_selected, picker
    if selection ~= first then
      message = label('The selection changed; structure edit cancelled.', '选择已变化，结构编辑已取消。', '選取已變更，結構編輯已取消。')
      return render('inspector')
    end
    local before, document = copy_order(owner.layers), recipe()
    local operations, desired, order = {}, {}, {}
    local can_plan = layer_edit_capability and actions.plan_layer_edit
    for index, item in ipairs(before) do
      if index >= first and index <= last then
        if not item.id then can_plan = false end
        operations[#operations + 1] = {action = 'remove', id = item.id}
      else desired[#desired + 1] = item end
    end
    local anchor = before[last + 1]
    for index, item in ipairs(inserted) do
      table.insert(desired, first + index - 1, item)
      visit_layers({item}, function(child) if not child.id then can_plan = false end end)
      operations[#operations + 1] = {action = 'add', slot = owner.name .. '_encoder',
        before = anchor and anchor.id or vim.NIL, layer = item}
    end
    for _, item in ipairs(desired) do
      if item.id then order[#order + 1] = item.id else can_plan = false end
    end
    if anchor and not anchor.id then can_plan = false end
    local function apply(plan)
      if not alive() or selected ~= owner or not vim.tbl_contains(draft.inputs, owner)
          or owner.layers ~= layers or layer_index ~= selection or branch_selected ~= selected_branch
          or picker ~= current_picker or not same_order(layers, before) or not vim.deep_equal(recipe(), document) then
        message = label('The draft changed while planning; structure edit cancelled.', '规划期间草稿已变化，结构编辑已取消。', '規劃期間草稿已變更，結構編輯已取消。')
        if alive() then render('inspector') end
        return
      end
      local result = plan and plan.outcomes[#plan.outcomes]
      if result and (result.slot ~= owner.name .. '_encoder' or result.id ~= inserted[#inserted].id
          or not vim.deep_equal(result.order, order)) then
        message = label('The backend structure differs; edit cancelled.', '后端结构不一致，编辑已取消。', '後端結構不一致，編輯已取消。')
        return render('inspector')
      end
      for _ = first, last do table.remove(layers, first) end
      for index = #inserted, 1, -1 do table.insert(layers, first, inserted[index]) end
      record_layer_list(owner, before, inserted[1])
      branch_selected, layer_index = nil, first
      local validation = plan and plan.validation
      if validation and validation.status == 'unavailable_or_invalid' then
        message = label('Structure updated; static generation needs attention: ', '结构已更新；静态生成需处理：', '結構已更新；靜態產生需處理：') .. tostring(validation.message)
      elseif plan then message = label('Structure planned; review and generate to save.', '结构已规划，审核生成后保存。', '結構已規劃，檢視產生後儲存。')
      else message = label('Structure edited locally; backend planning needs stable IDs.', '结构已在本地编辑；后端规划需要稳定 ID。', '結構已在本機編輯；後端規劃需要穩定 ID。') end
      render('inspector')
    end
    if can_plan and #operations <= 64 then
      message = label('Planning structure…', '正在规划结构…', '正在規劃結構…')
      render('inspector')
      actions.plan_layer_edit(document, operations, function(plan, error)
        if not alive() then return end
        if not plan then message = error; return render('inspector') end
        apply(plan)
      end)
    else apply(nil) end
  end
  local function fresh_layer_id()
    local temporary = {type = 'identity'}
    local ok, error = assign_detached_ids(temporary)
    if not ok then return nil, error end
    return temporary.id
  end
  plan_layer_copy = function(owner, source, targets)
    local selection, selected_branch, current_picker = layer_index, branch_selected, picker
    local document, reserved, changes, operations = recipe(), {}, {}, {}
    local total = 0
    for _, input in ipairs(draft.inputs) do total = total + count_layers(input.layers) end
    if total + #targets * count_layers({source}) > 64 then
      message = label('The copy would exceed the 64-layer model limit.', '复制后将超过模型的 64 层上限。', '複製後將超過模型的 64 層上限。')
      return render('inspector')
    end
    local can_plan = layer_cross_copy_capability and actions.plan_layer_edit and source.id ~= nil
    for _, target in ipairs(targets) do
      if not vim.tbl_contains(draft.inputs, target) or target.factory or target.native_output
          or #(target.layers or {}) >= 16 then
        message = label('A destination changed; copy cancelled.', '目标编码器已变化，复制已取消。', '目標編碼器已變更，複製已取消。')
        return render('inspector')
      end
      local copied = vim.deepcopy(source)
      visit_layers({copied}, function(item) item.id = nil end)
      local ok, error = assign_detached_ids(copied, reserved)
      if not ok then message = tostring(error); return render('inspector') end
      local ids = {}
      visit_layers({copied}, function(item) if item.id then ids[#ids + 1] = item.id else can_plan = false end end)
      local before = copy_order(target.layers or {})
      local order = {}
      for _, item in ipairs(before) do
        if item.id then order[#order + 1] = item.id else can_plan = false end
      end
      order[#order + 1] = copied.id
      changes[#changes + 1] = {owner = target, original = target.layers, before_present = target.layers ~= nil,
        before = before, copied = copied, copied_value = vim.deepcopy(copied), order = order}
      operations[#operations + 1] = {action = 'copy', id = source.id, to_slot = target.name .. '_encoder', new_ids = ids}
    end
    local function apply(plan)
      if not alive() or selected ~= owner or not vim.tbl_contains(draft.inputs, owner)
          or type(owner.layers) ~= 'table' or owner.layers[selection] ~= source
          or layer_index ~= selection or branch_selected ~= selected_branch or picker ~= current_picker
          or not vim.deep_equal(recipe(), document) then
        message = label('The draft changed while planning; copy cancelled.', '规划期间草稿已变化，复制已取消。', '規劃期間草稿已變更，複製已取消。')
        if alive() then render('inspector') end
        return
      end
      for i, change in ipairs(changes) do
        if not vim.tbl_contains(draft.inputs, change.owner) or change.owner.layers ~= change.original
            or (change.original and not same_order(change.original, change.before))
            or (plan and (plan.outcomes[i].id ~= change.copied.id
              or plan.outcomes[i].slot ~= change.owner.name .. '_encoder'
              or not vim.deep_equal(plan.outcomes[i].order, change.order))) then
          message = label('A destination or backend result changed; copy cancelled.', '目标或后端结果已变化，复制已取消。', '目標或後端結果已變更，複製已取消。')
          return render('inspector')
        end
      end
      for _, change in ipairs(changes) do
        change.owner.layers = change.owner.layers or {}
        change.owner.layers[#change.owner.layers + 1] = change.copied
        change.after = copy_order(change.owner.layers)
      end
      record_layer_batch(owner, source, changes)
      local validation = plan and plan.validation
      if validation and validation.status == 'unavailable_or_invalid' then
        message = label('Copied in draft; static generation needs attention: ', '已复制到草稿；静态生成需处理：', '已複製到草稿；靜態產生需處理：') .. tostring(validation.message)
      elseif plan then message = label('Backend planned all copies; undo restores them together.', '后端已规划全部复制；撤销时会一起恢复。', '後端已規劃全部複製；復原時會一起還原。')
      else message = label('Copied locally; backend planning needs a compatible CLI and stable IDs.', '已在本地复制；后端规划需要兼容的 CLI 和稳定 ID。', '已在本機複製；後端規劃需要相容的 CLI 和穩定 ID。') end
      render('inspector')
    end
    if can_plan then
      message = label('Planning copies…', '正在规划复制…', '正在規劃複製…')
      render('inspector')
      actions.plan_layer_edit(document, operations, function(plan, error)
        if not alive() then return end
        if not plan then message = error; return render('inspector') end
        apply(plan)
      end)
    else apply(nil) end
  end
  local function preview()
    if #draft.inputs == 0 or #draft.outputs == 0 then
      message = label('Add an input and an output head.', '请添加输入和输出头。', '請新增輸入和輸出頭。')
      render('inspector')
      return
    end
    for _, rows in ipairs({ draft.inputs, draft.outputs }) do
      local names = {}
      for _, row in ipairs(rows) do
        if names[row.name] then
          message = label('Each group needs unique names.', '每组条目的名称必须唯一。', '每組項目的名稱必須唯一。')
          render('inspector')
          return
        end
        names[row.name] = true
      end
    end
    local config = recipe()
    local fields = { inputs = {}, outputs = {}, routes = config.routes or {}, components = config.components or {},
      encoder_layers = config.encoder_layers or {},
      encoder_widths = config.encoder_widths or {},
      native_encoders = config.native_encoders or {},
      component_params = config.component_params or {},
      data = vim.deepcopy(draft.data),
      training = vim.deepcopy(config.training),
      supervision = vim.deepcopy(config.supervision),
      reconcile_assembly = draft.reconcile_assembly,
      base_revision = draft.base_revision }
    for _, row in ipairs(draft.inputs) do
      fields.inputs[#fields.inputs + 1] = row.name .. (row.dtype ~= '' and ':' .. row.dtype or '') .. '=' .. row.shape
    end
    for _, row in ipairs(draft.outputs) do
      if row.loss == 'ce' and tonumber(row.width) < 2 then
        select_node('output', row)
        message = label('Cross entropy needs at least two classes.', '交叉熵至少需要两个类别。', '交叉熵至少需要兩個類別。')
        return render('inspector')
      end
      fields.outputs[#fields.outputs + 1] = row.name .. '=' .. row.width .. ':' .. row.loss .. '@' .. row.weight
    end
    local chosen_layer = branch_selected or (kind == 'encoder' and selected and selected.layers and selected.layers[layer_index or 0])
    fields.selected_layer_id = chosen_layer and chosen_layer.id or nil
    close()
    actions.preview(fields)
  end
  local function choose_factory()
    picker, message = 'factory', nil
    if component_choices then return render('inspector') end
    render('inspector')
    actions.factories(function(choices)
      if not alive() then return end
      component_choices = choices
      if picker == 'factory' then render('inspector') end
    end)
  end
  local function set_factory(reference)
    local slot = kind == 'encoder' and selected.name .. '_encoder' or kind
    local existing = kind == 'encoder' and selected.factory or draft.components[kind]
    if existing ~= reference then draft.component_params[slot] = nil end
    if kind == 'encoder' then selected.factory = reference; if not reference then selected.component_width = nil end
    else draft.components[kind] = reference end
    picker = nil
    render('inspector')
  end
  local function add_layer()
    if selected.native_output then
      message = label('Switch output to flatten before adding a built-in layer stack.',
        '添加内置层序列前，请将输出方式切换为展平。', '新增內建層序列前，請將輸出方式切換為攤平。')
      return render('inspector')
    end
    local count = 0
    for _, input in ipairs(draft.inputs) do count = count + count_layers(input.layers) end
    if #(selected.layers or {}) >= 16 or count >= 64 then
      message = label('Layer limit reached (16 per encoder, 64 per model).', '层数已达上限（每个编码器 16 层，模型 64 层）。',
        '層數已達上限（每個編碼器 16 層，模型 64 層）。')
      return render('inspector')
    end
    picker, message, layer_query = 'layers', nil, ''
    render('inspector')
    if not layer_catalog then
      actions.layer_catalog(function(catalog, supports_factory, supports_residual, supports_parallel, supports_custom_loss, supports_layer_ids, supports_layer_plan, supports_layer_edit, supports_cross_copy)
        if not alive() then return end
        layer_catalog, factory_layers, residual_layers, parallel_layers, custom_losses = catalog, supports_factory, supports_residual, supports_parallel, supports_custom_loss
        id_capability = supports_layer_ids
        layer_plan_capability = supports_layer_plan
        layer_edit_capability = supports_layer_edit
        layer_cross_copy_capability = supports_cross_copy
        local _, _, has_ids = id_state()
        ids_enabled = id_capability and (not draft.base_revision or has_ids)
        if picker == 'layers' then render('inspector') end
      end)
    end
  end
  local function add_factory_layer(reference, signature)
    if not alive() or type(reference) ~= 'string' or #reference > 240 or not reference:match('^[^%s:;=@]+:[^%s:;=@]+$') then
      message = label('Invalid layer factory.', '层工厂无效。', '層工廠無效。'); render('inspector'); return
    end
    local target = selected
    vim.ui.input({ prompt = label('Output width (factory must return a PyTorch module; checked on run): ',
      '输出宽度（工厂须返回 PyTorch 模块，运行时验证）：', '輸出寬度（工廠須回傳 PyTorch 模組，執行時驗證）：') }, function(raw)
      if not alive() or selected ~= target or raw == nil then return end
      local width = tonumber(raw)
      if not width or width % 1 ~= 0 or width < 1 or width > 65536 then
        message = label('Invalid output width.', '输出宽度无效。', '輸出寬度無效。'); render('inspector'); return
      end
      selected.layers = selected.layers or {}
      local index = (layer_index or #selected.layers) + 1
      local layer = { type = 'factory', reference = reference, output_width = width, params = vim.empty_dict() }
      plan_layer_list_change(selected, selected.layers, index, 'add', layer, signature)
    end)
  end
  local function remove()
    if kind == 'input' then
      draft.component_params[selected.name .. '_encoder'] = nil
      for i, input in ipairs(draft.inputs) do
        if input == selected then table.remove(draft.inputs, i); break end
      end
      for _, input in ipairs(draft.inputs) do
        if input.source == selected then input.source = nil end
      end
    elseif kind == 'output' then
      for i, output in ipairs(draft.outputs) do
        if output == selected then table.remove(draft.outputs, i); break end
      end
    end
    select_node('fusion')
  end

  local render_helpers = {assign_layer_ids = assign_layer_ids, branch_groups = branch_groups, supported_layer = supported_layer, count_layers = count_layers, estimated_width = estimated_width, same_order = same_order, copy_order = copy_order, edit_layer = edit_layer, move_layer = move_layer, fresh_layer_id = fresh_layer_id, plan_layer_list_change = plan_layer_list_change, find_branch_layer = find_branch_layer}
  for name, value in pairs({valid_name = valid_name, parse_shape = parse_shape, cell = cell, id_state = id_state, assign_detached_ids = assign_detached_ids, select_node = select_node, connect_input = connect_input, keyboard_connect = keyboard_connect, undo_move = undo_move, edit_parameters = edit_parameters, example = example, classification = classification, sequence = sequence, preview = preview, choose_factory = choose_factory, set_factory = set_factory, add_layer = add_layer, add_factory_layer = add_factory_layer, remove = remove}) do render_helpers[name] = value end
  function render(focus)
    if ids_enabled then
      local ok, error = render_helpers.assign_layer_ids()
      if not ok then ids_enabled = false; message = label('Layer identity could not be assigned: ', '无法分配层身份：', '無法指派層身分：') .. tostring(error) end
    end
    if not alive() then return end
    local cursor = api.nvim_win_get_cursor(win)
    local lines, styles = {}, {}
    targets = {}
    local function emit(cells)
      local row, text = #lines + 1, ''
      for _, item in ipairs(cells) do
        local start = #text
        text = text .. item.text
        if item.style then styles[#styles + 1] = { row, start, #text, item.style } end
        if item.activate then targets[#targets + 1] = { row, start, #text, item.activate, item.drag } end
      end
      lines[row] = text
    end
    local function plain(text, style) emit({render_helpers.cell(text, width, style)}) end
    local function toolbar(buttons)
      local used, toolbar_row = 0, {}
      for _, item in ipairs(buttons) do
        local w = math.min(width, vim.fn.strdisplaywidth(item[1]) + 2)
        if used + w + 1 > width and used > 0 then emit(toolbar_row); used, toolbar_row = 0, {} end
        toolbar_row[#toolbar_row + 1] = render_helpers.cell(' ' .. item[1], w, item[3] or 'Pmenu', item[2])
        toolbar_row[#toolbar_row + 1] = render_helpers.cell(' ', 1)
        used = used + w + 1
      end
      if used > 0 then emit(toolbar_row) end
    end
    plain('  ◇ Model-Maven  /  ' .. label('Model canvas', '模型画布', '模型畫布'), 'Title')
    if draft.base_revision then plain('  model-maven.json  ·  ' .. label('Edit saved model', '编辑已保存模型', '編輯已儲存模型'), 'Comment') end
    if draft.base_revision and actions.assembly_status then
      local matches, reason = actions.assembly_status(draft)
      if matches == false then
        plain(label('  Project assembly at opening differs · load it or review your draft',
          '  打开时的项目装配与草稿不同 · 可载入并审核', '  開啟時的專案裝配與草稿不同 · 可載入並檢視'), 'WarningMsg')
      elseif matches == nil and reason then
        plain(label('  Project assembly cannot be compared: ', '  无法比较项目装配：', '  無法比較專案裝配：') .. tostring(reason), 'WarningMsg')
      end
    end
    if draft.base_revision and actions.data_status then
      local matches = actions.data_status(draft)
      if matches == false then
        plain(label('  Project input differs from declaration · load it before rebuilding',
          '  项目输入与声明不一致 · 重建前请载入', '  專案輸入與宣告不一致 · 重建前請載入'), 'WarningMsg')
      elseif matches == nil then
        plain(label('  Project input could not be compared with the declaration',
          '  无法核对项目输入与声明', '  無法核對專案輸入與宣告'), 'WarningMsg')
      end
    end
    if draft.base_revision and actions.training_status and actions.training_status(draft) == false then
      plain(label('  Last selected training settings differ from declaration · load them before rebuilding',
        '  最近选择的训练设置与声明不一致 · 重建前可载入', '  最近選擇的訓練設定與宣告不一致 · 重建前可載入'), 'WarningMsg')
    end
    toolbar({
      {label('+ Input', '+ 输入', '+ 輸入'), function() add('input') end},
      {label('+ Output', '+ 输出头', '+ 輸出頭'), function() add('output') end},
      {label('Review and generate', '审核并生成', '檢視並產生'), render_helpers.preview, 'PmenuSel'},
    })
    if actions.graph_canvas then
      toolbar({{label('Explicit graph…', '显式图…', '顯式圖…'), function()
        close(); actions.graph_canvas()
      end}})
    end
    if id_capability and not ids_enabled then
      local _, missing = render_helpers.id_state()
      if #missing > 0 then toolbar({{label('Enable stable layer IDs', '启用稳定层身份', '啟用穩定層身分'), function()
        local ok, error = render_helpers.assign_layer_ids()
        if ok then
          ids_enabled = true
          message = label('Layer IDs added to the draft; review and save to persist.',
            '已为草稿中的层分配身份，审核保存后生效。', '已為草稿中的層指派身分，檢視儲存後生效。')
        else message = label('Could not assign layer IDs: ', '无法分配层身份：', '無法指派層身分：') .. tostring(error) end
        render('inspector')
      end}}) end
    end
    if #move_undo > 0 or #move_redo > 0 then
      toolbar({
        {label('Undo layer change [u]', '撤销层修改 [u]', '復原層修改 [u]'), function() render_helpers.undo_move(false) end},
        {label('Redo layer change [Ctrl-R]', '重做层修改 [Ctrl-R]', '重做層修改 [Ctrl-R]'), function() render_helpers.undo_move(true) end},
      })
    end
    if draft.base_revision and actions.sync_assembly then
      toolbar({{label('Load saved assembly', '载入已保存装配', '載入已儲存裝配'), function()
        actions.sync_assembly(function(ok, err)
          if not alive() then return end
          message = ok and label('Saved assembly loaded. Review changes before generating.',
            '已载入项目装配，生成前请审核变化。', '已載入專案裝配，產生前請檢視變更。') or err
          render('inspector')
        end)
      end}})
    end
    if draft.base_revision and actions.sync_data then
      toolbar({{label('Load project input', '载入项目输入', '載入專案輸入'), function()
        actions.sync_data(function(data, err)
          if not alive() then return end
          if data then draft.data = data end
          message = data and label('Project input loaded; it will be saved in the declaration.',
            '已载入项目输入，将写入模型声明。', '已載入專案輸入，將寫入模型宣告。') or err
          render('inspector')
        end)
      end}})
    end
    if draft.base_revision and actions.edit_data then
      toolbar({{label('Edit project input', '编辑项目输入', '編輯專案輸入'), function()
        close()
        actions.edit_data()
      end}})
    end
    if draft.base_revision and actions.package_project then
      toolbar({{label('Package project', '打包项目', '打包專案'), function()
        actions.package_project()
      end}})
    end
    if draft.base_revision and actions.sync_training then
      toolbar({{label('Load training settings', '载入训练设置', '載入訓練設定'), function()
        actions.sync_training(function(training, err)
          if not alive() then return end
          if training then draft.training = training end
          message = training and label('Training settings loaded.', '已载入训练设置。', '已載入訓練設定。') or err
          render('inspector')
        end)
      end}})
    end
    if actions.link_supervision then
      toolbar({{label('Link supervision file', '关联监督文件', '連結監督檔案'), function()
        actions.link_supervision(function(reference, err)
          if not alive() then return end
          if reference then draft.supervision = reference end
          message = reference and label('Supervision reference saved in the draft; import and rebind in the target project.',
            '监督引用已加入草稿；目标项目需导入并重新绑定。', '監督參照已加入草稿；目標專案需匯入並重新繫結。') or err
          render('inspector')
        end)
      end}})
    end
    if draft.supervision then
      toolbar({{label('Remove supervision reference', '移除监督引用', '移除監督參照'), function()
        draft.supervision = nil
        message = label('Reference removed; saved rules are unchanged.', '引用已移除，已保存规则不变。', '參照已移除，已儲存規則不變。')
        render('inspector')
      end}})
    end
    if draft.training then
      toolbar({{label('Remove training declaration', '移除训练声明', '移除訓練宣告'), function()
        draft.training = nil; message = label('Training declaration removed.', '已移除训练声明。', '已移除訓練宣告。'); render('inspector')
      end}})
    end
    if draft.data then
      toolbar({{label('Remove input from declaration', '从声明移除输入', '從宣告移除輸入'), function()
        draft.data = nil
        message = label('Declaration cleared; project input kept.',
          '已移除声明，项目输入保留。', '已移除宣告，專案輸入保留。')
        render('inspector')
      end}})
    end
    plain('')
    local split = width >= 72
    local graph_width = split and math.floor(width * 0.61) or width
    local side_width = split and width - graph_width - 3 or width
    local canvas, inspector = {}, {}
    local function canvas_line(cells) canvas[#canvas + 1] = cells end
    local function section(text, style) canvas_line({render_helpers.cell(text, graph_width, style)}) end
    local function node(text, w, active, action)
      return render_helpers.cell(' ' .. text, w, active and 'PmenuSel' or 'Pmenu', action)
    end
    local card_width = math.floor((graph_width - 5) / 2)
    local empty = #draft.inputs == 0 and #draft.outputs == 0 and not next(draft.components)
    if empty then
      section(label(' Start designing your model', ' 开始设计你的模型', ' 開始設計你的模型'), 'Title')
      section(label(' Structure → data → run → improve', ' 结构 → 数据 → 运行 → 迭代', ' 結構 → 資料 → 執行 → 迭代'), 'Comment')
      section('')
      if actions.graph_canvas then
        canvas_line({ node(label('Open graph canvas', '打开自由图画布', '開啟自由圖畫布'), graph_width, true, function()
          close(); actions.graph_canvas()
        end) })
        section(actions.graph_exists and label(' Saved graph exists · reopen to continue', ' 已有图声明 · 打开后继续编辑', ' 已有圖宣告 · 開啟後繼續編輯')
          or label(' Blank canvas · branches, multiple inputs and custom operators',
            ' 空画布 · 分支、多输入和自定义算子', ' 空畫布 · 分支、多輸入和自訂運算子'), 'Comment')
        section('')
      end
      canvas_line({ node(label('Sequential structure', '顺序网络结构', '順序網路結構'), graph_width, not actions.graph_canvas, render_helpers.sequence) })
      section(label(' Input → add and arrange layers → output', ' 输入 → 添加并排列层 → 输出', ' 輸入 → 新增並排列層 → 輸出'), 'Comment')
      section('')
      canvas_line({ node(label('Start with a blank input', '从空白输入开始', '從空白輸入開始'), graph_width, false, function() add('input') end) })
      section(label(' Add your own inputs, layers and outputs', ' 自行添加输入、网络层和输出', ' 自行新增輸入、網路層和輸出'), 'Comment')
      section('')
      canvas_line({ node(label('Example · binary classifier', '示例 · 二分类模型', '範例 · 二元分類模型'), graph_width, false, render_helpers.classification) })
      section(label(' Vector input → Linear → ReLU → 2 classes', ' 向量输入 → Linear → ReLU → 两个类别',
        ' 向量輸入 → Linear → ReLU → 兩個類別'), 'Comment')
      section('')
      if actions.image_project then
        canvas_line({ node(label('Image render_helpers.classification project', '图像分类项目', '影像分類專案'), graph_width, false, function()
          close(); actions.image_project()
        end) })
        section(label(' Class folders → image input → editable model', ' 类别文件夹 → 图像输入 → 可编辑模型', ' 類別資料夾 → 影像輸入 → 可編輯模型'), 'Comment')
        section('')
      end
      canvas_line({ node(label('Example · two-input regression', '双输入回归示例', '雙輸入回歸範例'), graph_width, false, render_helpers.example) })
      section(label(' Vector and variable-length render_helpers.sequence → score', ' 向量和变长序列 → 分数',
        ' 向量和變長序列 → 分數'), 'Comment')
    else
    section(label(' Drag input to encoder · keyboard: input c → encoder c',
      ' 拖动输入到编码器 · 键盘：输入 c → 编码器 c', ' 拖曳輸入至編碼器 · 鍵盤：輸入 c → 編碼器 c'), 'Comment')
    if route_pending then
      section(label(' Connecting from ', ' 正在从此输入接线：', ' 正在從此輸入接線：') .. route_pending.source.name
        .. label(' · select encoder, press c · Esc cancels', ' · 选择编码器并按 c · Esc 取消', ' · 選擇編碼器並按 c · Esc 取消'), 'Search')
    end
    canvas_line({
      render_helpers.cell(label(' INPUT', ' 输入', ' 輸入'), card_width, 'Comment'), render_helpers.cell('', 5),
      render_helpers.cell(label(' ENCODER', ' 编码器', ' 編碼器'), card_width, 'Comment'),
    })
    if #draft.inputs == 0 then section(label(' Add an input to begin.', ' 从添加输入开始。', ' 從新增輸入開始。'), 'Comment') end
    for _, input in ipairs(draft.inputs) do
      local source = input.source or input
      local shape, dynamic_axes = render_helpers.parse_shape(source.shape or '')
      local input_width = 1
      local dynamic = {}
      for _, axis in ipairs(dynamic_axes or {}) do dynamic[axis + 1] = true end
      if shape then
        for axis, size in ipairs(shape) do if not dynamic[axis] then input_width = input_width * size end end
      else input_width = nil end
      local source_cell = node(source.name, card_width, kind == 'input' and selected == source, function() render_helpers.select_node('input', source) end)
      source_cell.drag = {route_source = source}
      local encoder_cell = node(input.name .. ' ↴' .. (input.layers and #input.layers > 0 and ('  ·  ' .. #input.layers) or ''),
        card_width, kind == 'encoder' and selected == input or (drag and drag.mode == 'route' and drag.target == input),
        function() render_helpers.select_node('encoder', input) end)
      encoder_cell.drag = {route_target = input}
      canvas_line({source_cell, render_helpers.cell(' ─→  ', 5, 'Comment'), encoder_cell})
      for index, layer in ipairs(input.layers or {}) do
        local active = kind == 'encoder' and selected == input and layer_index == index
        local marker = drag and drag.owner == input and drag.container == input.layers and drag.target == index and ' ⇅ ' or '   '
        local output_width = render_helpers.estimated_width(layer, input_width)
        local dimensions = input_width and output_width and ('  ' .. input_width .. ' → ' .. output_width) or '  ? → ?'
        local source_label = layer.type == 'factory' and ('  ' .. (layer.reference or '?')) or ''
        local item = render_helpers.cell(marker .. index .. '  ' .. (layer_names[layer.type] or tostring(layer.type))
          .. dimensions .. source_label, graph_width,
          drag and drag.owner == input and drag.container == input.layers and drag.target == index and 'Search' or (active and not branch_selected and 'PmenuSel' or 'Pmenu'), function()
            branch_selected = nil
            kind, selected, tab, picker, message, layer_index = 'encoder', input, 'properties', nil, nil, index
            render('inspector')
          end)
        item.drag = { owner = input, layer = layer, index = index, container = input.layers }
        canvas_line({ item })
        local function show_branch(parent, depth, path)
          for group_index, group in ipairs(render_helpers.branch_groups(parent)) do
          for child_index, child in ipairs(group) do
            local child_path = path .. (parent.type == 'parallel' and ('/' .. group_index) or '') .. '.' .. child_index
            local chosen = active and branch_selected == child
            local dragging = drag and drag.owner == input and drag.container == group and drag.target == child_index
            local child_cell = render_helpers.cell(string.rep('  ', math.min(depth, 8)) .. (dragging and ' ⇅ ' or ' ↳ ') .. child_path .. '  '
              .. (layer_names[child.type] or tostring(child.type)), graph_width, dragging and 'Search' or (chosen and 'PmenuSel' or 'Pmenu'), function()
                kind, selected, tab, picker, message, layer_index = 'encoder', input, 'properties', nil, nil, index
                branch_selected = child
                render('inspector')
              end)
            child_cell.drag = { owner = input, layer = child, index = child_index, container = group, root = layer, root_index = index }
            canvas_line({child_cell})
            show_branch(child, depth + 1, child_path)
          end
          end
        end
        show_branch(layer, 1, tostring(index))
        input_width = output_width
      end
    end
    -- A source may feed multiple encoders. Keep all declared inputs reachable.
    for _, input in ipairs(draft.inputs) do
      local connected = false
      for _, encoder in ipairs(draft.inputs) do
        if (encoder.source or encoder) == input then connected = true; break end
      end
      if not connected then
        local unconnected = node(input.name, card_width, kind == 'input' and selected == input, function() render_helpers.select_node('input', input) end)
        unconnected.drag = {route_source = input}
        canvas_line({unconnected, render_helpers.cell(label(' Unconnected', ' 未连接', ' 未連接'), graph_width - card_width, 'Comment')})
      end
    end
    section('  └────────────↓', 'Comment')
    canvas_line({node(label('Fusion', '融合', '融合'), card_width, kind == 'fusion', function() render_helpers.select_node('fusion') end),
      render_helpers.cell(' ─→  ', 5, 'Comment'),
      node(label('Output module', '输出模块', '輸出模組'), card_width, kind == 'head', function() render_helpers.select_node('head') end)})
    section(label(' ↓ OUTPUT HEADS', ' ↓ 输出头', ' ↓ 輸出頭'), 'Comment')
    for _, output in ipairs(draft.outputs) do
      canvas_line({node(output.name, card_width, kind == 'output' and selected == output, function() render_helpers.select_node('output', output) end),
        render_helpers.cell('  ' .. output.width .. ' · ' .. output.loss, graph_width - card_width, 'Comment')})
    end
    end
    local function detail(text, style, activate)
      inspector[#inspector + 1] = {render_helpers.cell(' ' .. text, side_width, style, activate)}
    end
    local function option(text, active, choose) detail((active and '● ' or '○ ') .. text, active and 'PmenuSel' or 'Pmenu', choose) end
    local properties, config = label('Properties', '属性', '屬性'), label('Config', '配置', '設定')
    local half = math.floor(side_width / 2)
    inspector[1] = {
      node(properties, half, tab == 'properties', function() tab, picker = 'properties', nil; render('inspector') end),
      node(config, side_width - half, tab == 'config', function() tab, picker = 'config', nil; render('inspector') end),
    }
    if not split then
      option(label('Back to canvas', '返回画布', '返回畫布'), false, function()
        api.nvim_win_set_cursor(win, {1, 0})
      end)
    end
    if message then detail(message, 'WarningMsg') end
    if tab == 'properties' and empty then
      detail(label('Choose a structure or start from a blank input.', '选择基础结构，或从空白输入开始。', '選擇基礎結構，或從空白輸入開始。'), 'Comment')
    elseif tab == 'config' then
      local draft_config = recipe()
      detail('model-maven.json', 'Title')
      detail('{')
      for _, key in ipairs({ 'version', 'data', 'training', 'supervision', 'inputs', 'outputs', 'routes', 'components', 'component_params', 'encoder_layers', 'encoder_widths', 'native_encoders' }) do
        local value = draft_config[key]
        if value ~= nil then
          if key == 'version' then
            detail('  "version": 1')
          elseif key == 'inputs' or key == 'outputs' then
            detail('  ' .. key, 'Title')
            for _, row in ipairs(value) do
              detail('    ' .. row.name, 'Identifier')
              if row.shape then
                detail('      shape: ' .. vim.json.encode(row.shape))
                detail('      dynamic_axes: ' .. vim.json.encode(row.dynamic_axes))
                detail('      dtype: ' .. vim.json.encode(row.dtype))
              else
                detail('      features: ' .. tostring(row.features))
                detail('      loss: ' .. row.loss)
                detail('      weight: ' .. tostring(row.weight))
              end
            end
          elseif type(value) == 'table' then
            detail('  ' .. key, 'Title')
            for _, slot in ipairs(vim.fn.sort(vim.tbl_keys(value))) do
              detail('    ' .. slot .. ' ← ' .. (type(value[slot]) == 'table' and vim.json.encode(value[slot]) or tostring(value[slot])))
            end
          end
        end
      end
      detail('}')
      detail(label('Exact JSON in review.', '审核页显示完整 JSON。', '檢視頁顯示完整 JSON。'), 'Comment')
    elseif picker == 'source' then
      detail(label('Connect input', '选择输入来源', '選擇輸入來源'), 'Title')
      for _, source in ipairs(draft.inputs) do
        option(source.name, (selected.source or selected) == source, function()
          render_helpers.connect_input(selected, source)
          picker = nil; render('inspector')
        end)
      end
      option(label('Back', '返回属性', '返回屬性'), false, function() picker = nil; render('inspector') end)
    elseif picker == 'factory' then
      detail(label('Component library', '组件库', '元件庫'), 'Title')
      option(label('Built-in default', '使用内置组件', '使用內建元件'), false, function() render_helpers.set_factory(nil) end)
      option(label('Back', '返回属性', '返回屬性'), false, function() picker = nil; render('inspector') end)
      if not component_choices then
        detail(label('Loading…', '正在读取…', '正在讀取…'), 'Comment')
      else
        detail(label('Candidates · not executed', '候选 · 尚未运行验证', '候選 · 尚未執行驗證'), 'Comment')
        for _, reference in ipairs(component_choices) do
          option(reference, false, function() render_helpers.set_factory(reference) end)
        end
        if #component_choices == 0 then detail(label('No project candidates.', '没有发现项目候选。', '沒有發現專案候選。'), 'Comment') end
      end
    elseif picker == 'layers' then
      detail(label('Add a layer', '添加网络层', '新增網路層'), 'Title')
      option(label('Search: ', '搜索：', '搜尋：') .. (layer_query ~= '' and layer_query or label('all', '全部', '全部')), false, function()
        vim.ui.input({ prompt = label('Search operators: ', '搜索算子：', '搜尋運算子：'), default = layer_query }, function(value)
          if not alive() or value == nil or picker ~= 'layers' then return end
          layer_query = vim.trim(value):lower()
          render('inspector')
        end)
      end)
      if not layer_catalog then detail(label('Loading…', '正在读取…', '正在讀取…'), 'Comment') end
      if layer_catalog and #layer_catalog == 0 then
        detail(label('Select a CLI that supports encoder layers.', '请选择支持网络层的 CLI。', '請選擇支援網路層的 CLI。'), 'WarningMsg')
      end
      local matches = 0
      for _, layer in ipairs(layer_catalog or {}) do
        if layer_query == '' or layer.label:lower():find(layer_query, 1, true) then
        matches = matches + 1
        option(layer.label, false, function()
          selected.layers = selected.layers or {}
          local index = (layer_index or #selected.layers) + 1
          render_helpers.plan_layer_list_change(selected, selected.layers, index, 'add', vim.deepcopy(layer.defaults))
        end)
        end
      end
      if layer_catalog and #layer_catalog > 0 and matches == 0 then
        detail(label('No matching built-in operators.', '没有匹配的内置算子。', '沒有符合的內建運算子。'), 'Comment')
      end
      if factory_layers then
        if actions.discover_module_operators then
          option(label('Discover operators in a Python module…', '发现 Python 模块中的算子…', '探索 Python 模組中的運算子…'), false, function()
            local target = selected
            actions.discover_module_operators(function(item)
              if not alive() or selected ~= target or picker ~= 'layers' then return end
              render_helpers.add_factory_layer(item.reference, item.signature)
            end)
          end)
        end
        if actions.discover_layers then
          option(label('Discover installed torch.nn operators…', '发现环境中的 torch.nn 算子…', '探索環境中的 torch.nn 運算子…'), false, function()
            local target = selected
            actions.discover_layers(function(layer)
              if not alive() or selected ~= target or picker ~= 'layers' then return end
              if type(layer) ~= 'table' or type(layer.name) ~= 'string' or not layer.name:match('^%a[%w_]*$') then return end
              render_helpers.add_factory_layer('torch.nn:' .. layer.name, layer.signature)
            end)
          end)
        end
        option(label('Choose project factory…', '选择项目工厂…', '選擇專案工廠…'), false, function()
          picker, message = 'layer_factory', nil
          render('inspector')
          if not component_choices then
            actions.factories(function(choices)
              if not alive() then return end
              component_choices = choices
              if picker == 'layer_factory' then render('inspector') end
            end)
          end
        end)
        option(label('Custom layer factory…', '自定义层工厂…', '自訂層工廠…'), false, function()
          vim.ui.input({ prompt = label('Layer factory module:callable: ', '层工厂 module:callable：', '層工廠 module:callable：') }, function(reference)
            if reference ~= nil then render_helpers.add_factory_layer(reference) end
          end)
        end)
      end
      option(label('Back', '返回属性', '返回屬性'), false, function() picker = nil; render('inspector') end)
    elseif picker == 'layer_factory' then
      detail(label('Project factories', '项目工厂', '專案工廠'), 'Title')
      detail(label('Static candidates · not verified', '静态候选 · 尚未验证', '靜態候選 · 尚未驗證'), 'Comment')
      if not component_choices then detail(label('Loading…', '正在读取…', '正在讀取…'), 'Comment')
      else
        local found = 0
        for _, reference in ipairs(component_choices) do
          if not reference:match('^component:[^@]+@') then
            found = found + 1
            option(reference, false, function() render_helpers.add_factory_layer(reference) end)
          end
        end
        if found == 0 then detail(label('No project candidates.', '没有发现项目候选。', '沒有發現專案候選。'), 'Comment') end
      end
      option(label('Back', '返回', '返回'), false, function() picker = 'layers'; render('inspector') end)
    elseif kind == 'input' then
      detail(selected.name, 'Title')
      option(label('Start connection [c]', '开始接线 [c]', '開始接線 [c]'), route_pending and route_pending.source == selected, render_helpers.keyboard_connect)
      local field = draft.data and draft.data.factory == 'model_csv:batch' and draft.data.params
        and draft.data.params.inputs and draft.data.params.inputs[selected.name]
      if type(field) == 'table' then
        local columns = field.columns or field
        if type(columns) == 'table' and vim.islist(columns) and #columns <= 256 then
          local names, valid = {}, true
          for _, name in ipairs(columns) do
            if type(name) ~= 'string' then valid = false; break end
            names[#names + 1] = vim.fn.strcharpart(name:gsub('[%c]', ' '), 0, 160)
          end
          if valid then detail(label('CSV input columns: ', 'CSV 输入列：', 'CSV 輸入欄：') .. table.concat(names, ', '), 'Comment') end
        end
      end
      if actions.selected_supervision then
        option(label('Inspect this input’s supervision…', '查看此输入的监督…', '查看此輸入的監督…'), false, function()
          local start, error = actions.selected_supervision('input', selected.name)
          if not start then message = error; return render('inspector') end
          close(); start()
        end)
      end
      if actions.forward_saved then
        option(label('Run saved model forward…', '运行已保存模型的前向…', '執行已儲存模型的前向…'), false, function()
          local start, error = actions.forward_saved()
          if not start then message = error; return render('inspector') end
          close(); start()
        end)
      end
      if actions.connect_csv then
        option(label('Connect CSV data…', '接入 CSV 数据…', '接入 CSV 資料…'), false, function()
          local name = selected.name
          close()
          actions.connect_csv(name)
        end)
      end
      if actions.edit_input_augmentation then
        detail(label('Project transforms save separately; external loops and evaluators must call context.transform_batch.',
          '项目变换单独保存；外部循环和评估器须调用 context.transform_batch。',
          '專案變換單獨儲存；外部迴圈和評估器須呼叫 context.transform_batch。'), 'Comment')
        option(label('Edit project transforms…', '编辑项目变换…', '編輯專案變換…'), false, function()
          local name = selected.name
          close()
          actions.edit_input_augmentation(name)
        end)
      end
      detail(label('Model data', '模型数据', '模型資料'), 'Title')
      detail(label('The batch factory supplies all inputs and targets.', '批次工厂为模型的全部输入和目标提供数据。', '批次工廠為模型全部輸入與目標提供資料。'), 'Comment')
      option(label('Data factory: ', '数据工厂：', '資料工廠：') .. (draft.data and draft.data.factory
        or label('synthetic until configured', '未配置时使用合成输入', '未設定時使用合成輸入')), false, function()
        local previous = draft.data
        vim.ui.input({ prompt = label('Batch factory module:callable: ', '批次工厂 module:callable：', '批次工廠 module:callable：'),
          default = previous and previous.factory or '' }, function(value)
          if not alive() or value == nil or draft.data ~= previous then return end
          value = vim.trim(value)
          if #value > 240 or not value:match('^[^%s:;=@]+:[^%s:;=@]+$') then
            message = label('Invalid factory reference.', '工厂引用无效。', '工廠參照無效。'); return render('inspector')
          end
          if previous and previous.factory == value then return end
          local function replace()
            if not alive() or draft.data ~= previous then return end
            draft.data = { factory = value, params = vim.empty_dict() }
            message = label('Data is declared in the draft; the factory has not been run.',
              '数据已写入草稿；尚未运行工厂。', '資料已寫入草稿；尚未執行工廠。')
            render('inspector')
          end
          if previous then
            vim.ui.select({label('Keep current data', '保留当前数据', '保留目前資料'), label('Replace factory and parameters', '替换工厂及参数', '取代工廠及參數')}, {
              prompt = label('Replace the current data declaration?', '替换当前数据声明？', '取代目前資料宣告？'),
            }, function(_, choice) if choice == 2 then replace() end end)
          else replace() end
        end)
      end)
      if draft.data then
        local data = draft.data
        detail(label('Factory parameters are saved in the declaration; do not enter secrets.',
          '工厂参数会保存到声明，请勿填写密钥。', '工廠參數會儲存到宣告，請勿填寫金鑰。'), 'Comment')
        option(label('Input binding: ', '输入绑定：', '輸入繫結：') .. (data.binding or label('automatic', '自动', '自動')), false, function()
          local modes = {'automatic', 'single', 'single-raw', 'args', 'kwargs', 'envelope'}
          vim.ui.select(modes, { prompt = label('Batch-to-model binding', '批次到模型的绑定方式', '批次到模型的繫結方式') }, function(_, index)
            if not alive() or draft.data ~= data or not index then return end
            if index == 1 then data.binding = nil else data.binding = modes[index] end
            render('inspector')
          end)
        end)
        option(label('Edit data factory parameters…', '编辑数据工厂参数…', '編輯資料工廠參數…'), false, function()
          render_helpers.edit_parameters(data.params, function(values) if draft.data == data then data.params = values end end)
        end)
        if data.factory == 'model_csv:batch' then
          local params = data.params or {}
          detail(label('CSV split: ', 'CSV 切分：', 'CSV 切分：') .. tostring(params.split or label('all rows', '全部行', '全部列')), 'Comment')
          option(label('Configure CSV split…', '配置 CSV 切分…', '設定 CSV 切分…'), false, function()
            local snapshot, port = vim.deepcopy(data), selected
            local function current() return alive() and selected == port and draft.data == data and vim.deep_equal(data, snapshot) end
            vim.ui.select({label('Use all rows', '使用全部行', '使用全部列'), 'train', 'val', 'test'}, {
              prompt = label('Choose split; the selected adapter must support split parameters.',
                '选择切分；所选适配器必须支持切分参数。', '選擇切分；所選配接器必須支援切分參數。'),
            }, function(_, choice)
              if not choice or not current() then return end
              local function apply(split, weights, seed)
                if not current() then message = label('Data changed; split not applied.', '数据已变化，未应用切分。', '資料已變更，未套用切分。'); if alive() then render('inspector') end; return end
                data.params = data.params or vim.empty_dict()
                data.params.split, data.params.split_weights, data.params.split_seed = split, weights, seed
                message = split and label('Split declared; render_helpers.preview the actual adapter. Row order changes membership.',
                  '已声明切分；请预览实际适配器。行顺序变化会改变归属。', '已宣告切分；請預覽實際配接器。列順序變更會改變歸屬。')
                  or label('Split removed; adapter will receive no split parameters.', '已移除切分参数。', '已移除切分參數。')
                render('inspector')
              end
              if choice == 1 then apply(nil, nil, nil); return end
              local old = type(params.split_weights) == 'table' and params.split_weights or {}
              local default = old.train ~= nil and old.val ~= nil and old.test ~= nil
                and (tostring(old.train) .. ',' .. tostring(old.val) .. ',' .. tostring(old.test)) or ''
              vim.ui.input({prompt = label('Integer weights for train,val,test (e.g. 8,1,1): ',
                'train,val,test 的整数权重（如 8,1,1）：', 'train,val,test 的整數權重（如 8,1,1）：'), default = default}, function(raw)
                if raw == nil or not current() then return end
                local a, b, c = raw:match('^%s*(%d+)%s*,%s*(%d+)%s*,%s*(%d+)%s*$')
                local train, val, test = tonumber(a), tonumber(b), tonumber(c)
                if not train or train > 1000000 or val > 1000000 or test > 1000000 or train + val + test == 0 then
                  message = label('Use three nonnegative integer weights with a positive sum.', '请输入三个非负整数权重，且总和大于零。', '請輸入三個非負整數權重，且總和大於零。'); return render('inspector')
                end
                vim.ui.input({prompt = label('Split seed (nonnegative integer): ', '切分种子（非负整数）：', '切分種子（非負整數）：'),
                  default = params.split_seed ~= nil and tostring(params.split_seed) or ''}, function(raw_seed)
                  if raw_seed == nil or not current() then return end
                  local seed = raw_seed:match('^%d+$') and tonumber(raw_seed) or nil
                  if not seed or seed > 9007199254740991 then
                    message = label('Enter an exactly representable nonnegative integer seed.', '请输入可精确表示的非负整数种子。', '請輸入可精確表示的非負整數種子。'); return render('inspector')
                  end
                  apply(({'train', 'val', 'test'})[choice - 1], {train = train, val = val, test = test}, seed)
                end)
              end)
            end)
          end)
        end
        if actions.preview_draft_data then
          option(label('Preview this batch…', '预览此批次…', '預覽此批次…'), false, function()
            local snapshot = vim.deepcopy(data)
            close()
            actions.preview_draft_data(snapshot)
          end)
        end
      end
      option(label('Name ', '名称 ', '名稱 ') .. selected.name, false, function()
        edit(label('Input name', '输入名称', '輸入名稱'), selected.name, render_helpers.valid_name, function(v)
          local old_slot, new_slot = selected.name .. '_encoder', v .. '_encoder'
          draft.component_params[new_slot] = draft.component_params[old_slot]
          if new_slot ~= old_slot then draft.component_params[old_slot] = nil end
          selected.name = v
        end)
      end)
      option(label('Shape ', '形状 ', '形狀 ') .. selected.shape, false, function()
        edit(label('Shape (e.g. 4,?8)', '形状（如 4,?8）', '形狀（如 4,?8）'), selected.shape,
          function(v) local _, _, normalized = render_helpers.parse_shape(v); return normalized end, function(v) selected.shape = v end)
      end)
      for _, item in ipairs({ {'4', label('Vector · [4]', '向量 · [4]', '向量 · [4]')},
          {'?8,4', label('Sequence · [?,4]', '序列 · [?,4]', '序列 · [?,4]')} }) do
        option(item[2], selected.shape == item[1], function() selected.shape = item[1]; render('inspector') end)
      end
      detail(label('Dtype', '数据类型', '資料類型'), 'Comment')
      for _, dtype in ipairs({ '', 'float32', 'int64' }) do
        option(dtype ~= '' and dtype or label('Runtime default', '运行时默认', '執行階段預設'), selected.dtype == dtype,
          function() selected.dtype = dtype; render('inspector') end)
      end
      option(label('Other dtype…', '其他类型…', '其他類型…'), false, function()
        edit(label('PyTorch dtype', 'PyTorch 数据类型', 'PyTorch 資料類型'), selected.dtype,
          function(v) if #v <= 64 then return v:match('^%a[%w_]*$') end end,
          function(v) selected.dtype = v end)
      end)
      option(label('Remove', '移除输入', '移除輸入'), false, render_helpers.remove)
    elseif kind == 'output' then
      detail(selected.name, 'Title')
      if actions.connect_csv then
        local target = draft.data and draft.data.factory == 'model_csv:batch' and draft.data.params
          and draft.data.params.targets and draft.data.params.targets[selected.name]
        if type(target) == 'table' then
          local columns = target.columns or target
          if type(columns) == 'table' and vim.islist(columns) and #columns <= 256 then
            local names, valid = {}, true
            for _, name in ipairs(columns) do
              if type(name) ~= 'string' then valid = false; break end
              names[#names + 1] = vim.fn.strcharpart(name:gsub('[%c]', ' '), 0, 160)
            end
            if valid then detail(label('CSV target columns: ', 'CSV 目标列：', 'CSV 目標欄：') .. table.concat(names, ', '), 'Comment') end
          end
        end
        option(label('Connect CSV targets…', '接入 CSV 目标…', '接入 CSV 目標…'), false, function()
          local name = selected.name
          close()
          actions.connect_csv(nil, name)
        end)
      end
      if actions.selected_supervision then
        option(label('Inspect this output’s supervision…', '查看此输出的监督…', '查看此輸出的監督…'), false, function()
          local start, error = actions.selected_supervision('output', selected.name)
          if not start then message = error; return render('inspector') end
          close(); start()
        end)
      end
      if actions.forward_saved then
        option(label('Run saved model forward…', '运行已保存模型的前向…', '執行已儲存模型的前向…'), false, function()
          local start, error = actions.forward_saved()
          if not start then message = error; return render('inspector') end
          close(); start()
        end)
      end
      option(label('Name ', '名称 ', '名稱 ') .. selected.name, false, function()
        edit(label('Head name', '输出头名称', '輸出頭名稱'), selected.name, render_helpers.valid_name, function(v) selected.name = v end)
      end)
      local min_width = selected.loss == 'ce' and 2 or 1
      option(label('Width ', '维度 ', '維度 ') .. selected.width, false, function()
        edit(label('Output dimension', '输出维度', '輸出維度'), selected.width, function(v)
          if v:match('^%d+$') and tonumber(v) >= min_width and tonumber(v) <= 65536 then return v end
        end, function(v) selected.width = v end)
      end)
      local half_width = math.floor(side_width / 2)
      local function adjust(delta)
        selected.width = tostring(math.max(min_width, math.min(65536, tonumber(selected.width) + delta)))
        render('inspector')
      end
      inspector[#inspector + 1] = {
        node(label('− 1', '− 1', '− 1'), half_width, false, function() adjust(-1) end),
        node(label('+ 1', '+ 1', '+ 1'), side_width - half_width, false, function() adjust(1) end),
      }
      option(label('Mean squared error', '均方误差', '均方誤差'), selected.loss == 'mse', function() selected.loss = 'mse'; render('inspector') end)
      option(label('Cross entropy', '交叉熵', '交叉熵'), selected.loss == 'ce', function()
        selected.loss = 'ce'
        selected.width = tostring(math.max(2, tonumber(selected.width)))
        render('inspector')
      end)
      if custom_losses then
        option(label('Custom loss function…', '自定义损失函数…', '自訂損失函數…'), selected.loss:match('^custom:') ~= nil, function()
          local output = selected
          vim.ui.input({ prompt = label('Loss function module:callable (prediction, target): ',
            '损失函数 module:callable（prediction, target）：', '損失函數 module:callable（prediction, target）：'),
            default = output.loss:match('^custom:(.*)$') or '' }, function(reference)
            if not alive() or selected ~= output or reference == nil then return end
            reference = vim.trim(reference)
            if #reference > 240 or not reference:match('^[^%s:;=@]+:[^%s:;=@]+$') then
              message = label('Invalid loss function reference.', '损失函数引用无效。', '損失函數參照無效。'); return render('inspector')
            end
            output.loss = 'custom:' .. reference
            message = label('Loss runs at training time; synthetic target shape is only an render_helpers.example.',
              '损失函数在训练时运行；合成目标形状仅为示例。', '損失函數在訓練時執行；合成目標形狀僅為範例。')
            render('inspector')
          end)
        end)
      end
      if selected.loss:match('^custom:') then
        detail(label('Custom: ', '自定义：', '自訂：') .. selected.loss:sub(8), 'Comment')
        detail(label('Used by the generated step; custom training loops must call it themselves.',
          '由生成的训练步骤使用；自定义训练循环需自行调用。', '由產生的訓練步驟使用；自訂訓練迴圈需自行呼叫。'), 'Comment')
      end
      option(label('Weight ', '权重 ', '權重 ') .. selected.weight, false, function()
        edit(label('Loss weight', '损失权重', '損失權重'), selected.weight, function(v)
          local n = tonumber(v); if n and n > 0 and n < math.huge then return tostring(n) end
        end, function(v) selected.weight = v end)
      end)
      detail(label('Training · shared by all outputs', '训练 · 所有输出共用', '訓練 · 所有輸出共用'), 'Title')
      detail(label('Targets come from the selected batch; binding is checked on execution.',
        '目标由所选批次提供，绑定在运行时检查。', '目標由所選批次提供，繫結在執行時檢查。'), 'Comment')
      option(label('Training entry: ', '训练入口：', '訓練入口：') .. (draft.training and (draft.training.mode .. ' · ' .. draft.training.entry)
        or label('not configured', '尚未配置', '尚未設定')), false, function()
        local previous, snapshot = draft.training, vim.deepcopy(draft.training)
        local choices = {
          label('Generated managed step', '生成的受管训练步骤', '產生的受管訓練步驟'),
          label('Custom managed step', '自定义受管训练步骤', '自訂受管訓練步驟'),
          label('Custom manual step', '自定义手动训练步骤', '自訂手動訓練步驟'),
          label('Custom external loop', '自定义外部训练循环', '自訂外部訓練循環'),
        }
        vim.ui.select(choices, { prompt = label('Choose training mode', '选择训练方式', '選擇訓練方式') }, function(_, choice)
          if not alive() or not choice or draft.training ~= previous or not vim.deep_equal(previous, snapshot) then return end
          local mode = ({'managed-step', 'managed-step', 'manual-step', 'external-loop'})[choice]
          local function install(entry)
            if not alive() or draft.training ~= previous or not vim.deep_equal(previous, snapshot) then return end
            local function apply()
              if not alive() or draft.training ~= previous or not vim.deep_equal(previous, snapshot) then return end
              draft.training = {mode = mode, entry = entry, params = vim.empty_dict()}
              message = label('Training declaration added; no training has run.', '训练声明已加入草稿；尚未训练。', '訓練宣告已加入草稿；尚未訓練。')
              render('inspector')
            end
            if previous then
              vim.ui.select({label('Keep current settings', '保留当前设置', '保留目前設定'), label('Replace settings', '替换设置', '取代設定')}, {
                prompt = label('Replace the existing training entry, parameters and optimizer?',
                  '替换现有训练入口、参数和优化器？', '取代現有訓練入口、參數與最佳化器？'),
              }, function(_, answer) if answer == 2 then apply() end end)
            else apply() end
          end
          if choice == 1 then install('$model:step'); return end
          vim.ui.input({ prompt = label('Training entry module:callable: ', '训练入口 module:callable：', '訓練入口 module:callable：') }, function(entry)
            if not alive() or entry == nil then return end
            entry = vim.trim(entry)
            if #entry > 240 or not entry:match('^[^%s:;=@]+:[^%s:;=@]+$') then
              message = label('Invalid training entry.', '训练入口无效。', '訓練入口無效。'); return render('inspector')
            end
            install(entry)
          end)
        end)
      end)
      if draft.training then
        local training = draft.training
        detail(label('Parameters are saved in the declaration; do not enter secrets.',
          '参数会保存到声明，请勿填写密钥。', '參數會儲存到宣告，請勿填寫金鑰。'), 'Comment')
        option(label('Edit training parameters…', '编辑训练参数…', '編輯訓練參數…'), false, function()
          render_helpers.edit_parameters(training.params, function(values) if draft.training == training then training.params = values end end)
        end)
        if training.mode ~= 'external-loop' then
          local optimizer = training.optimizer
          detail(label('Optimizer: ', '优化器：', '最佳化器：') .. (optimizer and optimizer.factory or 'AdamW'), 'Comment')
          if not optimizer or not optimizer.factory then
            for _, field in ipairs({
              {name = 'lr', title = label('Learning rate', '学习率', '學習率'), positive = true},
              {name = 'weight_decay', title = 'Weight decay'},
            }) do
              option(field.title .. ': ' .. tostring(optimizer and optimizer[field.name] or label('runtime default', '运行时默认', '執行階段預設')), false, function()
                local before = training.optimizer
                edit(field.title .. label(' (empty = default): ', '（留空使用默认）：', '（留空使用預設）：'),
                  before and before[field.name] and tostring(before[field.name]) or '', function(raw)
                    if raw == '' then return false end
                    local number = tonumber(raw)
                    if number and number < math.huge and (field.positive and number > 0 or not field.positive and number >= 0) then return number end
                  end, function(value)
                    if draft.training ~= training or training.optimizer ~= before then return end
                    training.optimizer = training.optimizer or {}
                    if value == false then training.optimizer[field.name] = nil else training.optimizer[field.name] = value end
                    if not next(training.optimizer) then training.optimizer = nil end
                  end)
              end)
            end
          end
          if training.mode == 'managed-step' then
            option(label('Custom optimizer factory…', '自定义优化器工厂…', '自訂最佳化器工廠…'), false, function()
              local before = training.optimizer
              vim.ui.input({ prompt = label('Factory module:callable (empty selects AdamW): ',
                '工厂 module:callable（留空选择 AdamW）：', '工廠 module:callable（留空選擇 AdamW）：'),
                default = before and before.factory or '' }, function(reference)
                if not alive() or draft.training ~= training or training.optimizer ~= before or reference == nil then return end
                reference = vim.trim(reference)
                if reference ~= '' and (#reference > 240 or not reference:match('^[^%s:;=@]+:[^%s:;=@]+$')) then
                  message = label('Invalid optimizer factory.', '优化器工厂无效。', '最佳化器工廠無效。'); return render('inspector')
                end
                if before and before.factory == reference then return end
                vim.ui.select({label('Keep optimizer', '保留优化器', '保留最佳化器'), label('Replace optimizer settings', '替换优化器设置', '取代最佳化器設定')}, {
                  prompt = label('Replace optimizer settings and parameters?', '替换优化器设置和参数？', '取代最佳化器設定與參數？'),
                }, function(_, choice)
                  if choice ~= 2 or not alive() or draft.training ~= training or training.optimizer ~= before then return end
                  training.optimizer = reference ~= '' and {factory = reference, params = vim.empty_dict()} or nil
                  render('inspector')
                end)
              end)
            end)
            if optimizer and optimizer.factory then
              option(label('Edit optimizer parameters…', '编辑优化器参数…', '編輯最佳化器參數…'), false, function()
                render_helpers.edit_parameters(optimizer.params, function(values)
                  if draft.training == training and training.optimizer == optimizer then optimizer.params = values end
                end)
              end)
            end
          end
        else detail(label('External loop owns optimizer and training behavior.', '外部循环负责优化器和训练行为。', '外部迴圈負責最佳化器與訓練行為。'), 'Comment') end
      end
      option(label('Remove', '移除输出头', '移除輸出頭'), false, render_helpers.remove)
    else
      local factory = kind == 'encoder' and selected.factory or draft.components[kind]
      if kind == 'encoder' then
        detail(selected.name .. '_encoder', 'Title')
        if route_pending then
          option(label('Connect from ', '连接输入 ', '連接輸入 ') .. route_pending.source.name .. ' [c]', false, render_helpers.keyboard_connect)
        end
        option(label('Source: ', '输入来源 ', '輸入來源 ') .. (selected.source or selected).name, false,
          function() picker = 'source'; render('inspector') end)
        if draft.base_revision and actions.edit_encoder_source then
          option(label('Plan source edits for this encoder…', '为此编码器规划源码修改…', '為此編碼器規劃原始碼修改…'), false, function()
            local name = selected.name
            close(); actions.edit_encoder_source(name)
          end)
        end
        option(label('Output: ', '输出方式：', '輸出方式：')
          .. (selected.native_output and label('native', '保留原始形状', '保留原始形狀') or label('flatten', '展平', '攤平')), false, function()
          if not selected.native_output and not selected.factory and selected.layers and #selected.layers > 0 then
            message = label('Native output needs a component or identity encoder; built-in layer stacks require flattened input.',
              '保留原始形状需要组件或恒等编码器；内置层序列需要展平输入。', '保留原始形狀需要元件或恆等編碼器；內建層序列需要攤平輸入。')
          else selected.native_output = not selected.native_output end
          render('inspector')
        end)
      end
      local default = kind == 'fusion' and label('Concatenate', '拼接', '拼接')
        or kind == 'head' and label('Linear heads', '线性输出头', '線性輸出頭')
        or label('Identity encoder', '恒等编码器', '恆等編碼器')
      local layers = kind == 'encoder' and (selected.layers or {}) or {}
      if kind == 'encoder' and not factory then
        local source_shape, axes = render_helpers.parse_shape((selected.source or selected).shape)
        local width = 1
        local dynamic = {}
        for _, axis in ipairs(axes) do dynamic[axis] = true end
        for axis, size in ipairs(source_shape) do if not dynamic[axis - 1] then width = width * size end end
        detail(label('Layers · input width ', '网络层 · 输入宽度 ', '網路層 · 輸入寬度 ') .. width, 'Title')
        detail(label('Drag a block or use [ / ] to move.', '拖动块或用 [ / ] 移动。', '拖曳區塊或用 [ / ] 移動。'), 'Comment')
        detail(label('u / Ctrl-R undo or redo layer changes.', 'u / Ctrl-R 撤销或重做层修改。', 'u / Ctrl-R 復原或重做層修改。'), 'Comment')
        option(label('+ Add layer', '+ 添加层', '+ 新增層'), false, render_helpers.add_layer)
        local layer = layers[layer_index]
        if layer then
          local owner = selected
          detail(label('Selected layer ', '选中层 ', '選取層 ') .. layer_index .. ' · '
            .. (layer_names[layer.type] or tostring(layer.type)), 'Title')
          local function adjust(field, by, min, max)
            local value = math.max(min, math.min(max, layer[field] + by))
            if field == 'probability' then value = tonumber(string.format('%.5f', value)) end
            plan_layer_value_change(owner, layer, layer, {[field] = value})
          end
          local function adjust_buttons(text, field, step, min, max)
            local left = math.floor(side_width / 2)
            inspector[#inspector + 1] = {
              node('− ' .. text, left, false, function() adjust(field, -step, min, max) end),
              node('+ ' .. text, side_width - left, false, function() adjust(field, step, min, max) end),
            }
          end
          if layer.type == 'factory' then
            option(label('Factory: ', '工厂：', '工廠：') .. layer.reference, false, function()
              edit(label('Layer factory module:callable', '层工厂 module:callable', '層工廠 module:callable'), layer.reference,
                function(raw) if #raw <= 240 and raw:match('^[^%s:;=]+:[^%s:;=]+$') then return raw end end,
                function(value) plan_layer_value_change(owner, layer, layer, {reference = value}) end)
            end)
            option(label('Output width: ', '输出宽度：', '輸出寬度：') .. layer.output_width, false, function()
              edit(label('Output width (verified at forward)', '输出宽度（前向时验证）', '輸出寬度（前向時驗證）'), tostring(layer.output_width),
                function(raw) local n = tonumber(raw); if n and n % 1 == 0 and n >= 1 and n <= 65536 then return n end end,
                function(value) plan_layer_value_change(owner, layer, layer, {output_width = value}) end)
            end)
            local signature = layer_signatures[layer]
            if type(signature) == 'table' and signature.status == 'observed' and type(signature.parameters) == 'table' then
              detail(label('Observed factory parameters (* required)', '已观察到的工厂参数（* 必填）', '已觀察到的工廠參數（* 必填）'), 'Title')
              for _, parameter in ipairs(signature.parameters) do
                local name = type(parameter) == 'table' and parameter.name or nil
                local kind = type(parameter) == 'table' and parameter.kind or nil
                if type(name) == 'string' and #name <= 128 and name:match('^[%a_][%w_]*$')
                    and (kind == 'POSITIONAL_OR_KEYWORD' or kind == 'KEYWORD_ONLY') then
                  local current = layer.params and layer.params[name]
                  local displayed = current ~= nil and vim.json.encode(current)
                    or label('(unset)', '（未设置）', '（未設定）')
                  option(name .. (parameter.required and ' *' or '') .. ' = ' .. displayed, false, function()
                    local owner, target_layer = selected, layer
                    vim.ui.input({ prompt = label('Value (JSON or plain text): ', '值（JSON 或普通文本）：', '值（JSON 或一般文字）：') .. name .. ' = ',
                      default = current ~= nil and vim.json.encode(current) or '' }, function(raw)
                      if not alive() or selected ~= owner or type(owner.layers) ~= 'table'
                          or owner.layers[layer_index] ~= target_layer or raw == nil then return end
                      local ok, value = pcall(vim.json.decode, raw)
                      if not ok then value = raw end
                      local params = vim.deepcopy(target_layer.params or vim.empty_dict())
                      params[name] = value
                      local encoded, serialized = pcall(vim.json.encode, params)
                      if not encoded or #serialized > 65536 or #vim.tbl_keys(params) > 64 then
                        message = label('Parameters are too large.', '参数过大。', '參數過大。')
                      else plan_layer_value_change(owner, target_layer, target_layer, {params = params}) end
                      render('inspector')
                    end)
                  end)
                elseif kind == 'POSITIONAL_ONLY' then
                  detail(tostring(name) .. ' · ' .. label('positional only; use a wrapper factory',
                    '仅位置参数；请使用包装工厂', '僅位置參數；請使用包裝工廠'), 'WarningMsg')
                end
              end
              if signature.truncated then detail(label('Signature truncated; other parameters may exist.',
                '签名已截断，可能还有其他参数。', '簽章已截斷，可能還有其他參數。'), 'WarningMsg') end
            else
              detail(label('Signature not observed; enter parameters manually.', '未观察到签名，请手动填写参数。', '未觀察到簽章，請手動填寫參數。'), 'Comment')
            end
            option(label('Edit parameters one by one…', '逐项编辑参数…', '逐項編輯參數…'), false, function()
              render_helpers.edit_parameters(layer.params, function(values, done)
                plan_layer_value_change(owner, layer, layer, {params = values}, done)
              end, true)
            end)
            option(label('Factory parameters…', '工厂参数…', '工廠參數…'), false, function()
              edit(label('Parameters as JSON object', '参数 JSON 对象', '參數 JSON 物件'), vim.json.encode(layer.params or vim.empty_dict()),
                function(raw)
                  local ok, value = pcall(vim.json.decode, raw)
                  if ok and #raw <= 65536 and raw:match('^%s*{') and type(value) == 'table' and #vim.tbl_keys(value) <= 64 then return value end
                end, function(value) plan_layer_value_change(owner, layer, layer, {params = next(value) and value or vim.empty_dict()}) end)
            end)
          elseif layer.type == 'residual' or layer.type == 'parallel' then
            detail(layer.type == 'residual' and label('Skip connection + branch', '跳连 + 分支', '跳接 + 分支')
              or label('Parallel branches', '并行分支', '平行分支'), 'Title')
            detail(layer.type == 'residual' and label('Branch width must match its input.', '分支输出宽度必须与输入一致。', '分支輸出寬度必須與輸入一致。')
              or label('Add needs equal widths; concat joins features.', '相加要求宽度一致；拼接合并特征。', '相加要求寬度一致；拼接合併特徵。'), 'Comment')
            if layer.type == 'parallel' then
              option(label('Merge: ', '汇合方式：', '匯合方式：') .. layer.merge, false, function()
                plan_layer_value_change(owner, layer, layer, {merge = layer.merge == 'add' and 'concat' or 'add'})
              end)
              if #layer.branches < 8 then option(label('+ Add branch', '+ 添加分支', '+ 新增分支'), false, function()
                local count = 0
                for _, input in ipairs(draft.inputs) do count = count + render_helpers.count_layers(input.layers) end
                if count >= 64 then message = label('Layer limit reached.', '层数已达上限。', '層數已達上限。'); return render('inspector') end
                local initial = {type = 'identity'}
                render_helpers.edit_layer(owner, layer, function() layer.branches[#layer.branches + 1] = {initial} end)
                branch_selected = initial
                render('inspector')
              end) end
            end
            local locate = render_helpers.find_branch_layer
            local function mutate_child(child, change)
              if selected ~= owner or not locate(layer, child) then return end
              render_helpers.edit_layer(owner, layer, change)
              render('inspector')
            end
            local function branch_options(parent, prefix)
              for group_index, group in ipairs(render_helpers.branch_groups(parent)) do
              for child_index, child in ipairs(group) do
                local child_path = prefix .. (parent.type == 'parallel' and ('/' .. group_index) or '') .. '.' .. child_index
                option(child_path .. ' · ' .. (layer_names[child.type] or tostring(child.type)), branch_selected == child, function()
                  branch_selected = child; render('inspector')
                end)
                if #render_helpers.branch_groups(child) > 0 then branch_options(child, child_path) end
              end
              end
            end
            branch_options(layer, tostring(layer_index))
            local child, parent, child_index = branch_selected
            if child then parent, child_index = locate(layer, child) end
            if child and not parent then branch_selected, child = nil, nil end
            if child then
              detail(label('Selected branch layer · ', '选中的分支层 · ', '選取的分支層 · ')
                .. (layer_names[child.type] or tostring(child.type)), 'Title')
              if child.features then
                option(label('Units ', '单元数 ', '單元數 ') .. child.features, false, function()
                  edit(label('Units', '单元数', '單元數'), tostring(child.features), function(raw)
                    local n = tonumber(raw); if n and n % 1 == 0 and n >= 1 and n <= 65536 then return n end
                  end, function(value) plan_layer_value_change(owner, layer, child, {features = value}) end)
                end)
              elseif child.probability then
                option(label('Drop probability ', '丢弃概率 ', '捨棄機率 ') .. child.probability, false, function()
                  edit(label('Drop probability', '丢弃概率', '捨棄機率'), tostring(child.probability), function(raw)
                    local n = tonumber(raw); if n and n >= 0 and n < 1 then return n end
                  end, function(value) plan_layer_value_change(owner, layer, child, {probability = value}) end)
                end)
              elseif child.type == 'factory' then
                option(label('Factory: ', '工厂：', '工廠：') .. child.reference, false, function()
                  edit(label('Layer factory module:callable', '层工厂 module:callable', '層工廠 module:callable'), child.reference,
                    function(raw) if #raw <= 240 and raw:match('^[^%s:;=]+:[^%s:;=]+$') then return raw end end,
                    function(value) plan_layer_value_change(owner, layer, child, {reference = value}) end)
                end)
                option(label('Output width: ', '输出宽度：', '輸出寬度：') .. child.output_width, false, function()
                  edit(label('Output width', '输出宽度', '輸出寬度'), tostring(child.output_width), function(raw)
                    local n = tonumber(raw); if n and n % 1 == 0 and n >= 1 and n <= 65536 then return n end
                  end, function(value) plan_layer_value_change(owner, layer, child, {output_width = value}) end)
                end)
                option(label('Edit parameters one by one…', '逐项编辑参数…', '逐項編輯參數…'), false, function()
                  render_helpers.edit_parameters(child.params, function(values, done)
                    plan_layer_value_change(owner, layer, child, {params = values}, done)
                  end, true)
                end)
              elseif child.type == 'parallel' then
                option(label('Merge: ', '汇合方式：', '匯合方式：') .. child.merge, false, function()
                  plan_layer_value_change(owner, layer, child, {merge = child.merge == 'add' and 'concat' or 'add'})
                end)
              end
              if child_index > 1 then option(label('↑ Move branch layer', '↑ 上移分支层', '↑ 上移分支層'), false, function()
                plan_layer_move(owner, parent, child_index, child_index - 1, layer)
              end) end
              if child_index < #parent then option(label('↓ Move branch layer', '↓ 下移分支层', '↓ 下移分支層'), false, function()
                plan_layer_move(owner, parent, child_index, child_index + 1, layer)
              end) end
              if #parent > 1 then option(label('Remove branch layer…', '移除分支层…', '移除分支層…'), false, function()
                vim.ui.select({ label('Keep layer', '保留层', '保留層'), label('Remove from draft', '从草稿移除', '從草稿移除') },
                  { prompt = label('Remove this branch layer and its parameters?', '移除此分支层及其参数？', '移除此分支層及其參數？') }, function(_, choice)
                    if choice ~= 2 or not alive() or selected ~= owner then return end
                    local current_parent, current_index = locate(layer, child)
                    if current_parent ~= parent or current_index ~= child_index or #parent <= 1 then return end
                    render_helpers.plan_layer_list_change(owner, parent, child_index, 'render_helpers.remove', child, nil, layer)
                  end)
              end) end
              if layer.type == 'parallel' and #layer.branches > 2 then
                local group_index
                for i, group in ipairs(layer.branches) do if group == parent then group_index = i; break end end
                if group_index then option(label('Remove selected branch…', '移除选中分支…', '移除選取分支…'), false, function()
                  vim.ui.select({label('Keep branch', '保留分支', '保留分支'), label('Remove branch', '移除分支', '移除分支')}, {
                    prompt = label('Remove this branch and all its layers?', '移除此分支及其所有层？', '移除此分支及其所有層？'),
                  }, function(_, choice)
                    if choice ~= 2 or not alive() or selected ~= owner or #layer.branches <= 2
                        or layer.branches[group_index] ~= parent or not locate(layer, child) then return end
                    render_helpers.edit_layer(owner, layer, function() table.remove(layer.branches, group_index) end)
                    branch_selected = nil
                    render('inspector')
                  end)
                end) end
              end
            end
            if layer_catalog then
              option(label('+ Add operator to branch…', '+ 在分支中添加算子…', '+ 在分支中新增運算子…'), false, function()
                local child_groups = child and render_helpers.branch_groups(child) or {}
                local destination = child_groups[1] or parent or render_helpers.branch_groups(layer)[1]
                local after = child and #child_groups == 0 and child_index or #destination
                local before, choices = render_helpers.copy_order(destination), {}
                for _, item in ipairs(layer_catalog) do choices[#choices + 1] = item.label end
                if #choices == 0 then
                  message = label('No supported operators are available.', '没有可用的算子。', '沒有可用的運算子。'); return render('inspector')
                end
                vim.ui.select(choices, { prompt = label('Add branch operator', '添加分支算子', '新增分支運算子') }, function(_, index)
                  if not alive() or not index or selected ~= owner or not vim.tbl_contains(draft.inputs, owner)
                      or not vim.tbl_contains(owner.layers or {}, layer) or not render_helpers.same_order(destination, before)
                      or (child and not locate(layer, child)) then return end
                  local count = 0
                  for _, input in ipairs(draft.inputs) do count = count + render_helpers.count_layers(input.layers) end
                  if #destination >= 16 or count >= 64 then
                    message = label('Layer limit reached.', '层数已达上限。', '層數已達上限。'); return render('inspector')
                  end
                  local item = layer_catalog[index]
                  if not item or not render_helpers.supported_layer(item.defaults) then return end
                  local added = vim.deepcopy(item.defaults)
                  render_helpers.plan_layer_list_change(owner, destination, after + 1, 'add', added, nil, layer)
                end)
              end)
            end
            if layer.type == 'residual' then option(label('Remove skip connection and unwrap…', '移除跳连并展开…', '移除跳接並展開…'), false, function()
              if #layers - 1 + #layer.branch > 16 then
                message = label('Unwrapping would exceed the 16-layer group limit.', '展开后超过每组 16 层的限制。', '展開後超過每組 16 層的限制。')
                return render('inspector')
              end
              local index, before, content = layer_index, render_helpers.copy_order(layers), vim.deepcopy(layer)
              vim.ui.select({label('Keep residual block', '保留残差块', '保留殘差區塊'),
                label('Remove skip and keep branch layers', '移除跳连并保留分支层', '移除跳接並保留分支層')}, {
                prompt = label('This changes the model: the input will no longer be added to the branch output.',
                  '这会改变模型：输入将不再与分支输出相加。', '這會改變模型：輸入將不再與分支輸出相加。'),
              }, function(_, choice)
                if choice ~= 2 or not alive() or selected ~= owner or layer_index ~= index
                    or owner.layers ~= layers or not render_helpers.same_order(layers, before) or layers[index] ~= layer
                    or not vim.deep_equal(layer, content) then return end
                plan_layer_structure(owner, index, index, render_helpers.copy_order(layer.branch))
              end)
            end) end
          elseif layer.features then
            option(label('Units ', '单元数 ', '單元數 ') .. layer.features, false, function()
              edit(label('Units', '单元数', '單元數'), tostring(layer.features), function(raw)
                local n = tonumber(raw)
                if n and n % 1 == 0 and n >= 1 and n <= 65536 then return n end
              end, function(n) plan_layer_value_change(owner, layer, layer, {features = n}) end)
            end)
            adjust_buttons('8', 'features', 8, 1, 65536)
          elseif layer.probability then
            option(label('Drop probability ', '丢弃概率 ', '捨棄機率 ') .. layer.probability, false, function()
              edit(label('Drop probability', '丢弃概率', '捨棄機率'), tostring(layer.probability), function(raw)
                local n = tonumber(raw)
                if n and n >= 0 and n < 1 then return n end
              end, function(n) plan_layer_value_change(owner, layer, layer, {probability = n}) end)
            end)
            adjust_buttons('5%', 'probability', 0.05, 0, 0.99)
          end
          local function move(offset)
            local target = layer_index + offset
            render_helpers.move_layer(selected, layer_index, target)
            render('inspector')
          end
          if layer_index > 1 then option(branch_selected and label('↑ Move outer block', '↑ 上移外层块', '↑ 上移外層區塊')
            or label('↑ Move layer up', '↑ 上移层', '↑ 上移層'), false, function() move(-1) end) end
          if layer_index < #layers then option(branch_selected and label('↓ Move outer block', '↓ 下移外层块', '↓ 下移外層區塊')
            or label('↓ Move layer down', '↓ 下移层', '↓ 下移層'), false, function() move(1) end) end
          if residual_layers then
            option(label('Wrap layers in a residual branch…', '将连续层包为残差分支…', '將連續層包為殘差分支…'), false, function()
              local index, before = layer_index, render_helpers.copy_order(layers)
              local count = 0
              for _, input in ipairs(draft.inputs) do count = count + render_helpers.count_layers(input.layers) end
              if count >= 64 then
                message = label('The residual would exceed the 64-layer limit.', '添加残差块将超过 64 层限制。', '新增殘差區塊將超過 64 層限制。')
                return render('inspector')
              end
              vim.ui.input({ prompt = label('Last layer number (inclusive): ', '结束层编号（包含）：', '結束層編號（包含）：'), default = tostring(index) }, function(raw)
                if not alive() or raw == nil or selected ~= owner or owner.layers ~= layers or not render_helpers.same_order(layers, before) then return end
                local last = tonumber(raw)
                if not last or last % 1 ~= 0 or last < index or last > #layers then
                  message = label('Choose a valid last layer.', '请选择有效的结束层。', '請選擇有效的結束層。'); return render('inspector')
                end
                local branch = {}
                for i = index, last do branch[#branch + 1] = layers[i] end
                local id, error = render_helpers.fresh_layer_id()
                if error then message = tostring(error); return render('inspector') end
                local block = { type = 'residual', id = id, branch = branch }
                if not render_helpers.supported_layer(block) then
                  message = label('Residual nesting exceeds the limit.', '残差嵌套超过限制。', '殘差巢狀超過限制。'); return render('inspector')
                end
                plan_layer_structure(owner, index, last, {block})
              end)
            end)
          end
          if parallel_layers then
            option(label('Create parallel branches from layer…', '将此层扩展为并行分支…', '將此層擴展為平行分支…'), false, function()
              local index, before = layer_index, render_helpers.copy_order(layers)
              local count = 0
              for _, input in ipairs(draft.inputs) do count = count + render_helpers.count_layers(input.layers) end
              if count + 2 > 64 then
                message = label('Parallel block would exceed the layer limit.', '并行块将超过层数限制。', '平行區塊將超過層數限制。'); return render('inspector')
              end
              vim.ui.select({label('Add outputs', '输出相加', '輸出相加'), label('Concatenate outputs', '输出拼接', '輸出拼接')}, {
                prompt = label('Choose branch merge:', '选择分支汇合方式：', '選擇分支匯合方式：'),
              }, function(_, choice)
                if not alive() or not choice or selected ~= owner or owner.layers ~= layers
                    or not render_helpers.same_order(layers, before) or layers[index] ~= layer then return end
                local id, error = render_helpers.fresh_layer_id()
                if error then message = tostring(error); return render('inspector') end
                local identity = {type = 'identity'}
                local ok, id_error = render_helpers.assign_detached_ids(identity, id and {[id] = true} or nil)
                if not ok then message = tostring(id_error); return render('inspector') end
                local block = {type = 'parallel', id = id, merge = choice == 1 and 'add' or 'concat',
                  branches = {{layer}, {identity}}}
                if not render_helpers.supported_layer(block) then
                  message = label('Parallel nesting exceeds the limit.', '并行嵌套超过限制。', '平行巢狀超過限制。'); return render('inspector')
                end
                plan_layer_structure(owner, index, index, {block})
              end)
            end)
          end
          option(label('Copy layer to other encoders…', '将此层复制到其他编码器…', '將此層複製到其他編碼器…'), false, function()
            local target_layer, origin = layer, selected
            local targets, names = {}, {}
            local count = 0
            for _, input in ipairs(draft.inputs) do
              count = count + render_helpers.count_layers(input.layers)
              if input ~= origin and not input.factory and not input.native_output and #(input.layers or {}) < 16 then
                targets[#targets + 1] = input
                names[#names + 1] = input.name .. '_encoder'
              end
            end
            if #targets == 0 then
              message = label('No compatible encoder has room for this layer.', '没有可接收此层的编码器。', '沒有可接收此層的編碼器。')
              return render('inspector')
            end
            local all_index
            if #targets > 1 then
              all_index = #names + 1
              names[all_index] = label('All compatible encoders', '所有可接收的编码器', '所有可接收的編碼器')
            end
            vim.ui.select(names, { prompt = label('Copy selected layer to:', '将选中层复制到：', '將選取的層複製到：') }, function(_, choice)
              if not alive() or selected ~= origin or not choice then return end
              local chosen = choice == all_index and targets or { targets[choice] }
              if not chosen[1] then return end
              if count + #chosen * render_helpers.count_layers({ target_layer }) > 64 then
                message = label('The copy would exceed the 64-layer model limit.', '复制后将超过模型的 64 层上限。', '複製後將超過模型的 64 層上限。')
              else
                plan_layer_copy(origin, target_layer, chosen)
              end
              render('inspector')
            end)
          end)
          option(label('Remove layer', '移除层', '移除層'), false, function()
            local owner, index, target_layer = selected, layer_index, layer
            vim.ui.select({ label('Keep layer', '保留层', '保留層'), label('Remove from draft', '从草稿移除', '從草稿移除') }, {
              prompt = label('Remove this layer and its parameters from the draft? Review before generating.',
                '从草稿移除此层及其参数？生成前仍可审核。', '從草稿移除此層及其參數？產生前仍可檢視。'),
            }, function(_, choice)
              if choice ~= 2 or not alive() or selected ~= owner or owner.layers ~= layers
                  or layers[index] ~= target_layer then return end
              render_helpers.plan_layer_list_change(owner, layers, index, 'render_helpers.remove', target_layer)
            end)
          end)
        end
        if #layers > 0 then detail(label('Layer list', '层列表', '層列表'), 'Title') end
        for index, listed_layer in ipairs(layers) do
          local after = render_helpers.estimated_width(listed_layer, width)
          local summary = listed_layer.probability and ('  ·  ' .. (listed_layer.probability * 100) .. '%')
            or ('  ·  ' .. tostring(width or '?') .. ' → ' .. tostring(after or '?'))
          option(index .. '  ' .. (layer_names[listed_layer.type] or tostring(listed_layer.type)) .. summary,
            layer_index == index, function() layer_index = index; render('inspector') end)
          width = after
        end
      end
      if #layers == 0 then
        detail(factory or default, 'Title')
        if factory then
          local slot = kind == 'encoder' and selected.name .. '_encoder' or kind
          option(label('Edit parameters one by one…', '逐项编辑参数…', '逐項編輯參數…'), false, function()
            render_helpers.edit_parameters(draft.component_params[slot], function(values) draft.component_params[slot] = values end)
          end)
          option(label('Factory parameters…', '工厂参数…', '工廠參數…'), false, function()
            local current = draft.component_params[slot] or vim.empty_dict()
            edit(label('Parameters as JSON object', '参数 JSON 对象', '參數 JSON 物件'),
              vim.json.encode(current), function(raw)
                local ok, values = pcall(vim.json.decode, raw)
                if ok and type(values) == 'table' and raw:match('^%s*{') and #raw <= 65536 then
                  for name in pairs(values) do
                    if type(name) ~= 'string' or #name > 128 or name == '' or name:find('[%c%s%.=]') then return nil end
                  end
                  return values
                end
              end, function(values) draft.component_params[slot] = next(values) and values or vim.empty_dict() end)
          end)
        end
        if kind == 'encoder' and factory then
          option(label('Copy component to other encoders…', '将组件复制到其他编码器…', '將元件複製到其他編碼器…'), false, function()
            local origin, reference = selected, factory
            local targets, names = {}, {}
            for _, input in ipairs(draft.inputs) do
              if input ~= origin and not input.factory and #(input.layers or {}) == 0 then
                targets[#targets + 1] = input
                names[#names + 1] = input.name .. '_encoder'
              end
            end
            if #targets == 0 then
              message = label('No empty encoder is available.', '没有空编码器可接收组件。', '沒有空編碼器可接收元件。')
              return render('inspector')
            end
            local all_index
            if #targets > 1 then
              all_index = #names + 1
              names[all_index] = label('All empty encoders', '所有空编码器', '所有空編碼器')
            end
            vim.ui.select(names, { prompt = label('Copy component and parameters to:', '复制组件及参数到：', '複製元件及參數到：') }, function(_, choice)
              if not alive() or selected ~= origin or not choice then return end
              local chosen = choice == all_index and targets or { targets[choice] }
              if not chosen[1] then return end
              local params = draft.component_params[origin.name .. '_encoder']
              for _, target in ipairs(chosen) do
                target.factory, target.component_width = reference, origin.component_width
                draft.component_params[target.name .. '_encoder'] = params and vim.deepcopy(params) or nil
              end
              message = label('Component copied. Review each encoder connection and width before generating.',
                '组件已复制，生成前请检查各编码器接线和宽度。', '元件已複製，產生前請檢查各編碼器接線和寬度。')
              render('inspector')
            end)
          end)
          option(label('Component output width ', '组件输出宽度 ', '元件輸出寬度 ')
            .. (selected.component_width and tostring(selected.component_width) or label('input width', '沿用输入宽度', '沿用輸入寬度')), false, function()
            edit(label('Output width (empty uses input width)', '输出宽度（留空沿用输入宽度）', '輸出寬度（留空沿用輸入寬度）'),
              selected.component_width and tostring(selected.component_width) or '', function(raw)
                if raw == '' then return false end
                local n = tonumber(raw)
                if n and n % 1 == 0 and n >= 1 and n <= 1000000 then return n end
              end, function(value) selected.component_width = value or nil end)
          end)
        end
        option(label('Choose component…', '选择组件…', '選擇元件…'), false, render_helpers.choose_factory)
        option(label('Enter own factory…', '指定自有工厂…', '指定自有工廠…'), false, function()
          edit(label('Module:callable', '模块:函数', '模組:函式'), factory or '',
            function(v) if #v <= 240 and v:match('^[^%s:;=@]+:[^%s:;=@]+$') then return v end end, render_helpers.set_factory)
        end)
      end
      if draft.base_revision and actions.assembly then
        option(label('Open component assembly…', '打开组件装配…', '開啟元件裝配…'), false, function()
          local slot = kind == 'encoder' and (selected.name .. '_encoder') or kind
          close()
          actions.assembly(slot)
        end)
      end
    end

    local inspector_row = split and #lines + 1 or #lines + #canvas + 3
    if split then
      for i = 1, math.max(#canvas, #inspector) do
        local cells = vim.list_extend({}, canvas[i] or {render_helpers.cell('', graph_width)})
        local used = 0
        for _, item in ipairs(cells) do used = used + vim.fn.strdisplaywidth(item.text) end
        cells[#cells + 1] = render_helpers.cell('', graph_width - used)
        cells[#cells + 1] = render_helpers.cell(' │ ', 3, 'Comment')
        vim.list_extend(cells, inspector[i] or {})
        emit(cells)
      end
    else
      for _, row in ipairs(canvas) do emit(row) end
      plain(''); plain(string.rep('─', width), 'Comment')
      for _, row in ipairs(inspector) do emit(row) end
    end
    plain('')
    local footer = {{label('Back to workspace', '返回工作区', '返回工作區'), close}}
    if not draft.base_revision then table.insert(footer, 1,
      {label('Advanced setup', '高级创建', '進階建立'), function() close(); actions.advanced() end}) end
    toolbar(footer)
    vim.bo[buf].modifiable = true
    api.nvim_buf_set_lines(buf, 0, -1, false, lines)
    vim.bo[buf].modifiable = false
    api.nvim_buf_clear_namespace(buf, ns, 0, -1)
    for _, span in ipairs(styles) do
      api.nvim_buf_set_extmark(buf, ns, span[1] - 1, span[2], { end_col = span[3], hl_group = span[4] })
    end
    if focus == 'inspector' and not split then cursor = { inspector_row, 0 } end
    local row = math.min(cursor[1], #lines)
    api.nvim_win_set_cursor(win, { row, math.min(cursor[2], #lines[row]) })
    if focus == 'inspector' and not split then api.nvim_win_call(win, function() vim.cmd('normal! zt') end) end
    if actions.checkpoint_draft then actions.checkpoint_draft(false) end
  end

  local function hit(row, column)
    for _, button in ipairs(targets) do
      if button[1] == row and column >= button[2] and column < button[3] then button[4](); return end
    end
  end
  local function target_at(row, column)
    for _, target in ipairs(targets) do
      if target[1] == row and column >= target[2] and column < target[3] then return target end
    end
  end
  local function drag_destination(moving, target)
    local info = target and target[5]
    if not info then return nil end
    if moving.mode == 'route' then return info.route_target end
    return info.owner == moving.owner and info.container == moving.container and info.index or nil
  end
  -- Keep long layer lists usable without requiring a single-screen network.
  local scroll_timer = (vim.uv or vim.loop).new_timer()
  local scroll_direction = 0
  local function stop_drag_scroll()
    scroll_direction = 0
    if scroll_timer then scroll_timer:stop() end
  end
  local function scroll_drag(direction)
    if not scroll_timer or scroll_direction == direction then return end
    stop_drag_scroll()
    if direction == 0 then return end
    scroll_direction = direction
    scroll_timer:start(250, 100, vim.schedule_wrap(function()
      if not drag or not alive() or scroll_direction == 0 then stop_drag_scroll(); return end
      local view = api.nvim_win_call(win, vim.fn.winsaveview)
      local last_top = math.max(1, api.nvim_buf_line_count(buf) - api.nvim_win_get_height(win) + 1)
      local top = math.max(1, math.min(last_top, view.topline + scroll_direction))
      if top == view.topline then return end
      view.topline = top
      view.lnum = math.min(api.nvim_buf_line_count(buf), top + drag.viewport_row - 1)
      view.col = drag.column
      api.nvim_win_call(win, function() vim.fn.winrestview(view) end)
      local target = target_at(view.lnum, drag.column)
      drag.target = drag_destination(drag, target)
      render()
    end))
  end
  local function next_button(direction)
    local row, col = unpack(api.nvim_win_get_cursor(win))
    local index
    for i, button in ipairs(targets) do
      if button[1] == row and col >= button[2] and col < button[3] then index = i; break end
    end
    if #targets > 0 then
      local target = targets[index and ((index + direction - 1) % #targets + 1) or 1]
      api.nvim_win_set_cursor(win, {target[1], target[2]})
    end
  end
  local function on_click()
    local mouse = vim.fn.getmousepos()
    if mouse.winid == win and mouse.line > 0 and mouse.column > 0 then
      api.nvim_win_set_cursor(win, {mouse.line, mouse.column - 1})
      local target = target_at(mouse.line, mouse.column - 1)
      if target and target[5] then
        local info = target[5]
        if info.route_source then
          route_pending = nil
          select_node('input', info.route_source)
          drag = {mode = 'route', source = info.route_source, document = recipe(), moved = false}
          return
        elseif info.route_target then
          drag = nil; hit(mouse.line, mouse.column - 1); return
        end
        branch_selected = info.root and info.layer or nil
        kind, selected, tab, picker, message, layer_index = 'encoder', info.owner, 'properties', nil, nil, info.root_index or info.index
        render()
        drag = { owner = info.owner, layer = info.layer, container = info.container, root = info.root,
          from = info.index, target = info.index, moved = false }
        return
      end
      drag = nil
      hit(mouse.line, mouse.column - 1)
    end
  end
  for _, key in ipairs({ '<LeftMouse>', '<2-LeftMouse>', '<3-LeftMouse>', '<4-LeftMouse>' }) do
    vim.keymap.set('n', key, on_click, { buffer = buf, nowait = true })
  end
  vim.keymap.set('n', '<LeftDrag>', function()
    if not drag or not alive() then return end
    local mouse = vim.fn.getmousepos()
    if mouse.winid ~= win then stop_drag_scroll(); drag.target = nil; render(); return end
    local view = api.nvim_win_call(win, vim.fn.winsaveview)
    drag.viewport_row = math.max(1, mouse.line - view.topline + 1)
    drag.column = math.max(0, mouse.column - 1)
    local height = api.nvim_win_get_height(win)
    scroll_drag(drag.viewport_row <= 1 and -1 or (drag.viewport_row >= height and 1 or 0))
    local target = mouse.winid == win and target_at(mouse.line, mouse.column - 1) or nil
    local destination = drag_destination(drag, target)
    drag.moved = true
    if drag.target ~= destination then drag.target = destination; render() end
  end, { buffer = buf, nowait = true })
  vim.keymap.set('n', '<LeftRelease>', function()
    if not drag or not alive() then return end
    local moving, mouse = drag, vim.fn.getmousepos()
    stop_drag_scroll()
    local view = api.nvim_win_call(win, vim.fn.winsaveview)
    local line = moving.viewport_row and view.topline + moving.viewport_row - 1 or mouse.line
    local column = moving.column or mouse.column - 1
    local target = mouse.winid == win and target_at(line, column) or nil
    local destination = drag_destination(moving, target)
    drag = nil
    if moving.mode == 'route' then
      if moving.moved and destination and vim.deep_equal(recipe(), moving.document)
          and vim.tbl_contains(draft.inputs, moving.source) and vim.tbl_contains(draft.inputs, destination) then
        connect_input(destination, moving.source)
        kind, selected, layer_index, branch_selected = 'encoder', destination, nil, nil
      elseif moving.moved then
        message = label('Connection cancelled. Drop on an encoder in the unchanged draft.',
          '接线已取消，请放到未变化草稿中的编码器上。', '接線已取消，請放到未變更草稿中的編碼器上。')
      end
      render(moving.moved and 'inspector' or nil)
      return
    end
    local layers = moving.container
    if moving.moved and destination and type(layers) == 'table' and layers[moving.from] == moving.layer
        and destination >= 1 and destination <= #layers then
      if moving.root then
        local parent = find_branch_layer(moving.root, moving.layer)
        if parent == layers then
          plan_layer_move(moving.owner, layers, moving.from, destination, moving.root)
        end
      else move_layer(moving.owner, moving.from, destination) end
    elseif moving.moved then
      message = label('Move cancelled. Drop on a layer in the same group.',
        '移动已取消，请放到同一层组内。', '移動已取消，請放到同一層組內。')
    end
    render(not moving.moved and width < 72 and 'inspector' or nil)
  end, { buffer = buf, nowait = true })
  vim.keymap.set('n', '<CR>', function() hit(unpack(api.nvim_win_get_cursor(win))) end, { buffer = buf })
  vim.keymap.set('n', '<Tab>', function() next_button(1) end, { buffer = buf })
  vim.keymap.set('n', '<S-Tab>', function() next_button(-1) end, { buffer = buf })
  for key, delta in pairs({ ['['] = -1, [']'] = 1, ['<M-Up>'] = -1, ['<M-Down>'] = 1 }) do
    vim.keymap.set('n', key, function()
      if kind ~= 'encoder' or not selected or not layer_index then return end
      if branch_selected then
        local root = selected.layers and selected.layers[layer_index]
        local parent, index = find_branch_layer(root, branch_selected)
        if parent and parent[index + delta] then
          plan_layer_move(selected, parent, index, index + delta, root)
        end
      elseif move_layer(selected, layer_index, layer_index + delta) then render('inspector') end
    end, { buffer = buf, nowait = true, desc = 'Move selected layer' })
  end
  vim.keymap.set('n', 'c', keyboard_connect, { buffer = buf, nowait = true, desc = 'Connect input to encoder' })
  vim.keymap.set('n', 'u', function() undo_move(false) end, { buffer = buf, nowait = true })
  vim.keymap.set('n', '<C-r>', function() undo_move(true) end, { buffer = buf, nowait = true })
  vim.keymap.set('n', 'q', close, { buffer = buf, nowait = true })
  vim.keymap.set('n', '<Esc>', function()
    if drag then
      local route = drag.mode == 'route'
      stop_drag_scroll(); drag = nil
      message = route and label('Connection cancelled.', '接线已取消。', '接線已取消。')
        or label('Move cancelled.', '移动已取消。', '移動已取消。')
      render()
    elseif route_pending then
      route_pending = nil; message = label('Connection cancelled.', '接线已取消。', '接線已取消。'); render()
    else close() end
  end, { buffer = buf, nowait = true })
  local group = api.nvim_create_augroup('ModelMavenBuilderDrag' .. buf, { clear = true })
  api.nvim_create_autocmd({ 'VimResized', 'WinLeave' }, { group = group, callback = function()
    if drag then stop_drag_scroll(); drag = nil; if alive() then render() end end
  end })
  api.nvim_create_autocmd('BufWipeout', { group = group, buffer = buf, once = true, callback = function()
    if actions.checkpoint_draft then actions.checkpoint_draft(true) end
    stop_drag_scroll()
    if scroll_timer and not scroll_timer:is_closing() then scroll_timer:close() end
    pcall(api.nvim_del_augroup_by_id, group)
  end })
  api.nvim_create_autocmd('VimLeavePre', { group = group, callback = function()
    if actions.checkpoint_draft then actions.checkpoint_draft(true) end
  end })
  if actions.layer_catalog then
    actions.layer_catalog(function(catalog, supports_factory, supports_residual, supports_parallel, supports_custom_loss, supports_layer_ids, supports_layer_plan, supports_layer_edit, supports_cross_copy)
      if not alive() then return end
      layer_catalog, factory_layers, residual_layers, parallel_layers, custom_losses = catalog, supports_factory, supports_residual, supports_parallel, supports_custom_loss
      id_capability = supports_layer_ids
      layer_plan_capability = supports_layer_plan
      layer_edit_capability = supports_layer_edit
        layer_cross_copy_capability = supports_cross_copy
      local _, _, has_ids = id_state()
      ids_enabled = id_capability and (not draft.base_revision or has_ids)
      render()
    end)
  end
  render()
end

M.supported_layer = supported_layer
return M
