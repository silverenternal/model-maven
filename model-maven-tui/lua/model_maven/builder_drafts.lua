-- Recovery files are separate from project declarations and owned per session.
local M, uv = {}, vim.uv
local limit = 4 * 1024 * 1024
local snapshot_limit = 1024 * 1024
local function graph_id(value)
  return type(value) == 'string' and #value > 0 and #value <= 128
    and not value:find('[^A-Za-z0-9_%-\128-\255]') and pcall(vim.str_utfindex, value)
end

local function directory(project, create)
  local path = vim.fn.stdpath('state')
  local base = uv.fs_lstat(path)
  if not base and create then
    vim.fn.mkdir(path, 'p', 448)
    base = uv.fs_lstat(path)
  end
  if not base then return nil end
  if base.type ~= 'directory' then return nil, 'Editor state directory is unavailable or is a link.' end
  for _, name in ipairs({'model-maven-drafts', vim.fn.sha256(project)}) do
    path = path .. '/' .. name
    local stat, err, code = uv.fs_lstat(path)
    if not stat and code == 'ENOENT' and create then
      local ok, failure = uv.fs_mkdir(path, 448)
      if not ok then stat = uv.fs_lstat(path); if not stat then return nil, failure end end
      stat = stat or uv.fs_lstat(path)
    elseif not stat and code ~= 'ENOENT' then return nil, err end
    if not stat then return nil end
    if stat.type ~= 'directory' then return nil, 'Draft directory is not a normal directory.' end
  end
  return path
end

local function read(path)
  local stat, err, code = uv.fs_lstat(path)
  if not stat then if code == 'ENOENT' then return nil end; return nil, err end
  if stat.type ~= 'file' or stat.size > limit then return nil, 'Draft is not a normal file or is too large.' end
  local fd, failure = uv.fs_open(path, 'r', 0)
  if not fd then return nil, failure end
  local raw, read_error = uv.fs_read(fd, limit + 1, 0)
  uv.fs_close(fd)
  if not raw or #raw > limit then return nil, read_error or 'Draft is too large.' end
  return raw
end

function M.snapshot(draft)
  local indices, inputs = {}, {}
  for i, input in ipairs(draft.inputs or {}) do indices[input] = i end
  for i, input in ipairs(draft.inputs or {}) do
    local copy = {}
    for key, value in pairs(input) do if key ~= 'source' then copy[key] = vim.deepcopy(value) end end
    if input.source then
      if not indices[input.source] then return nil, 'Draft input refers to a missing source.' end
      copy.source_index = indices[input.source]
    end
    inputs[i] = copy
  end
  local payload = {version = 1, inputs = inputs, outputs = vim.deepcopy(draft.outputs or {}),
    components = vim.deepcopy(draft.components or {}), component_params = vim.deepcopy(draft.component_params or {}),
    data = vim.deepcopy(draft.data), training = vim.deepcopy(draft.training), supervision = vim.deepcopy(draft.supervision),
    base_revision = draft.base_revision, reconcile_assembly = draft.reconcile_assembly}
  local ok, raw = pcall(vim.json.encode, payload)
  if not ok or #raw > snapshot_limit then return nil, 'Cannot encode draft or it is too large.' end
  return raw
end

function M.pack(draft_raw, base_raw)
  local ok, raw = pcall(vim.json.encode, {version = 2, draft = draft_raw, base = base_raw})
  if not ok or #raw > limit then return nil, 'Cannot encode recovery envelope.' end
  return raw
end

-- Merge only whole, independently changed sections. A structure edit and a
-- data/training edit can refer to the same ports, so do not merge those across
-- sides without a human resolving the dependency.
function M.merge(base_raw, draft_raw, saved_raw, validate)
  if not base_raw then return nil, {'base unavailable'} end
  local values = {}
  for _, raw in ipairs({base_raw, draft_raw, saved_raw}) do
    local ok, value = pcall(vim.json.decode, raw)
    local valid, recovered = pcall(validate, raw)
    if not ok or type(value) ~= 'table' or value.version ~= 1 or not valid or not recovered then
      return nil, {'invalid snapshot'}
    end
    values[#values + 1] = value
  end
  local base, mine, theirs = unpack(values)
  if base.base_revision ~= mine.base_revision then return nil, {'base revision mismatch'} end
  local structural = {'inputs', 'outputs', 'components', 'component_params', 'reconcile_assembly'}
  local configuration = {'data', 'training', 'supervision'}
  local function changed(left, right, fields)
    for _, field in ipairs(fields) do if not vim.deep_equal(left[field], right[field]) then return true end end
    return false
  end
  if (changed(base, mine, structural) and changed(base, theirs, configuration))
      or (changed(base, theirs, structural) and changed(base, mine, configuration)) then
    return nil, {'structure and data/training/supervision dependencies'}
  end
  if (changed(base, mine, {'inputs'}) and changed(base, theirs, {'components', 'component_params'}))
      or (changed(base, theirs, {'inputs'}) and changed(base, mine, {'components', 'component_params'}))
      or (changed(base, mine, {'components'}) and changed(base, theirs, {'component_params'}))
      or (changed(base, theirs, {'components'}) and changed(base, mine, {'component_params'})) then
    return nil, {'input/component dependencies'}
  end
  local merged, conflicts, changes = vim.deepcopy(theirs), {}, {}
  for _, field in ipairs({'inputs', 'outputs', 'components', 'component_params', 'reconcile_assembly', 'data', 'training', 'supervision'}) do
    local mine_changed = not vim.deep_equal(base[field], mine[field])
    local theirs_changed = not vim.deep_equal(base[field], theirs[field])
    if mine_changed and theirs_changed and not vim.deep_equal(mine[field], theirs[field]) then
      conflicts[#conflicts + 1] = field
    elseif mine_changed then
      merged[field] = vim.deepcopy(mine[field])
      changes[#changes + 1] = field
    elseif theirs_changed then changes[#changes + 1] = field end
  end
  if #conflicts > 0 then return nil, conflicts end
  local ok, merged_raw = pcall(vim.json.encode, merged)
  if not ok or #merged_raw > snapshot_limit then return nil, {'merged snapshot too large'} end
  local valid, result = pcall(validate, merged_raw)
  if not valid or not result then return nil, {'merged snapshot invalid'} end
  return result, nil, changes
end

function M.create(project, kind)
  kind = kind or 'builder'
  if kind ~= 'builder' and kind ~= 'graph' then return nil, 'Invalid recovery kind.' end
  local root, err = directory(project, true)
  if not root then return nil, err end
  local bytes, failure = uv.random(16)
  if not bytes then return nil, failure end
  local id = bytes:gsub('.', function(char) return string.format('%02x', string.byte(char)) end)
  return {path = root .. '/' .. kind .. '-' .. id .. '.json', root = root}
end

function M.write(handle, raw)
  if type(raw) ~= 'string' or #raw > limit then return nil, 'Draft is too large.' end
  local previous, err = read(handle.path)
  if err then return nil, err end
  if (handle.digest and (not previous or vim.fn.sha256(previous) ~= handle.digest)) or (not handle.digest and previous) then
    return nil, 'Recovery file changed outside this session.'
  end
  if previous == raw then return true end
  local bytes, random_error = uv.random(12)
  if not bytes then return nil, random_error end
  local temp = handle.path .. '.' .. bytes:gsub('.', function(char) return string.format('%02x', string.byte(char)) end) .. '.tmp'
  local fd, open_error = uv.fs_open(temp, 'wx', 384)
  if not fd then return nil, open_error end
  local written, write_error = uv.fs_write(fd, raw, 0)
  local synced, sync_error
  if written == #raw then synced, sync_error = uv.fs_fsync(fd) end
  uv.fs_close(fd)
  if not synced then uv.fs_unlink(temp); return nil, write_error or sync_error or 'Draft write failed.' end
  local check, check_error = read(handle.path)
  if check_error or (handle.digest and (not check or vim.fn.sha256(check) ~= handle.digest)) or (not handle.digest and check) then
    uv.fs_unlink(temp); return nil, check_error or 'Recovery file changed during save.'
  end
  local renamed, rename_error = uv.fs_rename(temp, handle.path)
  if not renamed then uv.fs_unlink(temp); return nil, rename_error end
  handle.digest = vim.fn.sha256(raw)
  if vim.fn.has('win32') ~= 1 then
    local dir, dir_error = uv.fs_open(handle.root, 'r', 0)
    if not dir then return nil, dir_error end
    local ok, sync_error = uv.fs_fsync(dir)
    uv.fs_close(dir)
    if not ok then return nil, sync_error end
  end
  return true
end

-- Display-only order is stored outside the project and never changes a graph.
function M.graph_layout(project)
  local root, err = directory(project, true)
  if not root then return nil, nil, err end
  local handle = {root = root, path = root .. '/graph-layout.json'}
  local raw, failure = read(handle.path)
  if failure then return nil, nil, failure end
  if not raw then return handle, {}, nil, {}, {} end
  local ok, payload = pcall(vim.json.decode, raw)
  if not ok or type(payload) ~= 'table' or (payload.version ~= 1 and payload.version ~= 2 and payload.version ~= 3) or type(payload.order) ~= 'table'
      or not vim.islist(payload.order) or #payload.order > 512 then return nil, nil, 'Invalid graph layout.' end
  local seen = {}
  for _, id in ipairs(payload.order) do
    if not graph_id(id) or seen[id] then
      return nil, nil, 'Invalid graph layout node ID.'
    end
    seen[id] = true
  end
  local collapsed = {}
  if payload.collapsed ~= nil then
    if type(payload.collapsed) ~= 'table' or not vim.islist(payload.collapsed) or #payload.collapsed > 512 then
      return nil, nil, 'Invalid collapsed group state.'
    end
    for _, id in ipairs(payload.collapsed) do
      if not graph_id(id) or collapsed[id] then
        return nil, nil, 'Invalid collapsed group ID.'
      end
      collapsed[id] = true
    end
  end
  local positions = {}
  if payload.version >= 2 then
    if type(payload.positions) ~= 'table' or (next(payload.positions) and vim.islist(payload.positions))
        or vim.tbl_count(payload.positions) > 512 then return nil, nil, 'Invalid graph positions.' end
    for id, point in pairs(payload.positions) do
      if not graph_id(id)
          or type(point) ~= 'table' or type(point.x) ~= 'number' or type(point.y) ~= 'number'
          or point.x % 1 ~= 0 or point.y % 1 ~= 0 or point.x < 0 or point.x > 2048 or point.y < 0 or point.y > 2048
          or vim.tbl_count(point) ~= 2 then return nil, nil, 'Invalid graph position.' end
      positions[id] = {x = point.x, y = point.y}
    end
  end
  local view
  if payload.version >= 2 and payload.view ~= nil and payload.view ~= vim.NIL then
    local value = payload.view
    if type(value) ~= 'table' or type(value.topline) ~= 'number' or type(value.leftcol) ~= 'number'
        or value.topline % 1 ~= 0 or value.leftcol % 1 ~= 0 or value.topline < 1 or value.topline > 4096
        or value.leftcol < 0 or value.leftcol > 4096 or (value.selected ~= nil and value.selected ~= vim.NIL
          and not graph_id(value.selected)) then
      return nil, nil, 'Invalid graph viewport.'
    end
    view = {topline = value.topline, leftcol = value.leftcol, selected = value.selected ~= vim.NIL and value.selected or nil}
  end
  handle.digest = vim.fn.sha256(raw)
  return handle, payload.order, nil, collapsed, positions, view
end

function M.write_graph_layout(handle, order, collapsed, positions, view)
  if type(order) ~= 'table' or not vim.islist(order) or #order > 512 then return nil, 'Invalid graph layout.' end
  local seen = {}
  for _, id in ipairs(order) do
    if not graph_id(id) or seen[id] then return nil, 'Invalid graph layout node ID.' end
    seen[id] = true
  end
  local groups = {}
  for id, value in pairs(collapsed or {}) do
    if value == true then
      if not graph_id(id) then return nil, 'Invalid collapsed group ID.' end
      groups[#groups + 1] = id
    end
  end
  if #groups > 512 then return nil, 'Too many collapsed groups.' end
  table.sort(groups)
  if type(positions) ~= 'table' or vim.tbl_count(positions) > 512 then return nil, 'Invalid graph positions.' end
  for id, point in pairs(positions) do
    if not graph_id(id)
        or type(point) ~= 'table' or type(point.x) ~= 'number' or type(point.y) ~= 'number'
        or point.x % 1 ~= 0 or point.y % 1 ~= 0 or point.x < 0 or point.x > 2048 or point.y < 0 or point.y > 2048
        or vim.tbl_count(point) ~= 2 then return nil, 'Invalid graph position.' end
  end
  if view ~= nil and (type(view) ~= 'table' or type(view.topline) ~= 'number' or type(view.leftcol) ~= 'number'
      or view.topline % 1 ~= 0 or view.leftcol % 1 ~= 0 or view.topline < 1 or view.topline > 4096
      or view.leftcol < 0 or view.leftcol > 4096 or (view.selected ~= nil and not graph_id(view.selected))) then return nil, 'Invalid graph viewport.' end
  local ok, raw = pcall(vim.json.encode, {version = 2, order = order, collapsed = groups,
    positions = next(positions) and positions or vim.empty_dict(), view = view})
  if not ok then return nil, 'Cannot encode graph layout.' end
  return M.write(handle, raw)
end

function M.list(project, kind)
  kind = kind or 'builder'
  if kind ~= 'builder' and kind ~= 'graph' then return {}, 'Invalid recovery kind.' end
  local root, err = directory(project, false)
  if not root then return {}, err end
  local scan, failure = uv.fs_scandir(root)
  if not scan then return {}, failure end
  local found = {}
  local recovery_kind = kind
  while true do
    local name, kind = uv.fs_scandir_next(scan)
    if not name then break end
    if kind == 'file' and name:match('^' .. recovery_kind .. '%-%x+%.json$') and #name == #recovery_kind + 38 then
      local raw = read(root .. '/' .. name)
      if raw then
        local ok, payload = pcall(vim.json.decode, raw)
        if ok and type(payload) == 'table' and (payload.version == 1 or (payload.version == 2 and type(payload.draft) == 'string')) then
          local stat = uv.fs_lstat(root .. '/' .. name)
          found[#found + 1] = {name = name, raw = payload.version == 2 and payload.draft or raw,
            base = payload.version == 2 and type(payload.base) == 'string' and payload.base or nil,
            modified = stat and stat.type == 'file' and stat.mtime and stat.mtime.sec or nil}
          if #found >= 64 then break end
        end
      end
    end
  end
  table.sort(found, function(a,b) return a.name < b.name end)
  return found
end
return M
