-- Explicit graph editing inside the existing workspace. Runtime validation,
-- generation and file persistence are delegated to the selected CLI.
local M = {}
local api = vim.api

local function valid_id(value)
  return type(value) == 'string' and #value > 0 and #value <= 128
    and not value:find('[^A-Za-z0-9_%-\128-\255]') and pcall(vim.str_utfindex, value)
end
function M.open(document, actions)
  local tr = actions.tr
  local width, offset = require('model_maven.model_preview').layout(110, actions.preview_boundary)
  local height = math.min(34, vim.o.lines - 5)
  if width < 36 or height < 9 then
    vim.notify(tr('Enlarge the terminal to edit the graph.', '请放大终端以编辑图。', '請放大終端以編輯圖。'), vim.log.levels.WARN)
    return
  end
  local buf = api.nvim_create_buf(false, true)
  vim.b[buf].model_maven_graph_window = true
  vim.bo[buf].bufhidden, vim.bo[buf].swapfile = 'wipe', false
  local win = api.nvim_open_win(buf, true, {relative = 'editor', style = 'minimal', border = 'rounded',
    width = width, height = height, row = 1, col = offset})
  vim.wo[win].wrap = false
  local ns = api.nvim_create_namespace('ModelMavenGraphCanvas')
  local selected, selected_edge, pending, dragging, message, generation = nil, nil, nil, nil, nil, 0
  local saved_history = type(actions.history) == 'table' and actions.history or nil
  local matching_history = saved_history and vim.deep_equal(saved_history.document, document)
    and vim.deep_equal(saved_history.layout, actions.layout_order or {})
    and vim.deep_equal(saved_history.positions, actions.layout_positions or {})
  local undo = matching_history and vim.deepcopy(saved_history.undo) or {}
  local redo = matching_history and vim.deepcopy(saved_history.redo) or {}
  local rows, edge_rows, port_hits, selected_row, selected_edge_row = {}, {}, {}, nil, nil
  local toolbar_rows = {}
  local group_rows, collapsed_groups = {}, vim.deepcopy(actions.collapsed_groups or {})
  local observed_signatures = {}
  local input_drafts = actions.input_drafts or {}
  local metadata_context = actions.metadata_context and actions.metadata_context() or nil
  local layout_order = vim.deepcopy(actions.layout_order or {})
  local layout_positions = vim.deepcopy(actions.layout_positions or {})
  local plane_view = vim.deepcopy(actions.layout_view)
  local focus_root = nil
  local render, plane_refresh, plane_status, plane_show_source, code_layout, source_on_right = nil, nil, nil, nil, false, false
  local function alive() return api.nvim_win_is_valid(win) and api.nvim_win_get_buf(win) == buf end
  local function show_source_side(source_right)
    if not alive() then return end
    code_layout, source_on_right = true, source_right == true
    local next_width = math.max(1, math.floor((vim.o.columns - 5) / 2))
    local next_height = math.max(1, math.min(34, vim.o.lines - 5))
    pcall(api.nvim_win_set_config, win, {relative = 'editor', row = 1,
      col = source_on_right and 1 or math.max(0, vim.o.columns - next_width - 2), width = next_width, height = next_height})
    if plane_show_source then plane_show_source(source_on_right) end
    render()
  end
  local function find_node(id)
    for _, node in ipairs(document.nodes) do if node.id == id then return node end end
  end
  local function find_definition(id)
    for _, definition in ipairs(document.definitions) do if definition.id == id then return definition end end
  end
  local function refresh_signatures()
    local current_context = actions.metadata_context and actions.metadata_context() or nil
    if current_context ~= metadata_context then observed_signatures, metadata_context = {}, current_context end
  end
  local function replace(next_doc)
    for key in pairs(document) do document[key] = nil end
    for key, value in pairs(vim.deepcopy(next_doc)) do document[key] = value end
  end
  local function sync_layout()
    local present, seen, result = {}, {}, {}
    for _, node in ipairs(document.nodes) do present[node.id] = true end
    for _, id in ipairs(layout_order) do
      if present[id] and not seen[id] then result[#result + 1] = id; seen[id] = true end
    end
    for _, node in ipairs(document.nodes) do
      if not seen[node.id] then result[#result + 1] = node.id; seen[node.id] = true end
    end
    layout_order = result
    for id in pairs(layout_positions) do if not present[id] then layout_positions[id] = nil end end
  end
  sync_layout()
  local function snapshot()
    return {document = vim.deepcopy(document), layout = vim.deepcopy(layout_order), positions = vim.deepcopy(layout_positions)}
  end
  local function save_history()
    if actions.history_changed then
      actions.history_changed({document = vim.deepcopy(document), layout = vim.deepcopy(layout_order),
        positions = vim.deepcopy(layout_positions), undo = vim.deepcopy(undo), redo = vim.deepcopy(redo)})
    end
  end
  local function layout_changed()
    if actions.layout_changed then actions.layout_changed(vim.deepcopy(layout_order), vim.deepcopy(collapsed_groups), vim.deepcopy(layout_positions)) end
  end
  local function refresh_plane()
    if plane_refresh then
      local ok, active = pcall(plane_refresh, vim.deepcopy(document), vim.deepcopy(layout_positions), generation)
      if not ok then vim.notify(tr('Could not update the planar view.', '无法更新平面视图。', '無法更新平面檢視。'), vim.log.levels.WARN) end
      if not ok or not active then plane_refresh = nil end
    end
  end
  local function mutate(fn)
    local before = snapshot()
    fn()
    sync_layout()
    local doc_changed = not vim.deep_equal(before.document, document)
    local layout_changed = not vim.deep_equal(before.layout, layout_order) or not vim.deep_equal(before.positions, layout_positions)
    if doc_changed or layout_changed then
      undo[#undo + 1] = before; if #undo > 64 then table.remove(undo, 1) end
      redo, generation, pending, dragging, selected_edge = {}, generation + 1, nil, nil, nil
      if doc_changed and actions.changed then actions.changed() end
      if layout_changed then
        if actions.layout_changed then actions.layout_changed(vim.deepcopy(layout_order), vim.deepcopy(collapsed_groups), vim.deepcopy(layout_positions)) end
      end
      save_history()
    end
    render()
    if doc_changed or layout_changed then refresh_plane() end
  end
  local function apply_edits(operations, build_candidate, done)
    local revision, base, candidate = generation, vim.deepcopy(document), vim.deepcopy(document)
    build_candidate(candidate)
    local function accept(proposal, error, warning)
      if not alive() or generation ~= revision or not vim.deep_equal(document, base) then return end
      if error or not proposal then
        message = error or tr('Edit planning failed.', '编辑规划失败。', '編輯規劃失敗。'); render()
        if plane_status then plane_status(message, selected) end
        return
      end
      if not vim.deep_equal(candidate, proposal) then
        message = tr('Backend proposal differs; no edit was applied.', '后端方案不一致，未应用修改。', '後端方案不一致，未套用修改。'); render(); return
      end
      mutate(function() replace(proposal) end)
      if done then done(warning) end
      if warning then message = warning; render(); if plane_status then plane_status(message, selected) end end
    end
    if actions.plan_edits then actions.plan_edits(base, operations, accept)
    else accept(nil, tr('Graph edit planning is unavailable; no change was applied.', '图编辑规划不可用，未应用修改。', '圖編輯規劃無法使用，未套用修改。')) end
  end
  local function patch_node(node, changes, done)
    apply_edits({{action = 'update_node', id = node.id, changes = changes}}, function(candidate)
      for _, item in ipairs(candidate.nodes) do
        if item.id == node.id then
          for key, value in pairs(changes) do
            if value == vim.NIL then item[key] = nil else item[key] = vim.deepcopy(value) end
          end
          break
        end
      end
    end, done)
  end
  local function patch_graph(changes, done)
    local function apply(candidate)
      for key, value in pairs(changes) do
        if value == vim.NIL then candidate[key] = nil else candidate[key] = vim.deepcopy(value) end
      end
    end
    if actions.plan_edits and actions.edit_graph_settings then
      apply_edits({{action = 'update_graph', changes = changes}}, apply, done)
    else
      message = tr('Graph settings planning is unavailable; no change was applied.', '图设置规划不可用，未应用修改。', '圖設定規劃無法使用，未套用修改。'); render()
    end
  end
  local function restore(reverse)
    local from, to = reverse and redo or undo, reverse and undo or redo
    if #from == 0 then message = tr('No change to restore.', '没有可恢复的修改。', '沒有可還原的修改。'); return render() end
    local before = snapshot()
    to[#to + 1] = before
    local restored = table.remove(from)
    replace(restored.document); layout_order = vim.deepcopy(restored.layout); layout_positions = vim.deepcopy(restored.positions or {}); sync_layout()
    generation, pending, dragging, selected_edge = generation + 1, nil, nil, nil
    save_history()
    if selected and not find_node(selected) then selected = nil end
    if not vim.deep_equal(before.document, document) and actions.changed then actions.changed() end
    if not vim.deep_equal(before.layout, layout_order) or not vim.deep_equal(before.positions, layout_positions) then layout_changed() end
    message = tr('Draft restored.', '草稿已恢复。', '草稿已還原。'); render(); refresh_plane()
  end
  local function input_id(prompt, default, done)
    local revision = generation
    vim.ui.input({prompt = prompt, default = default}, function(value)
      if not alive() or generation ~= revision or value == nil then return end
      value = vim.trim(value)
      if not valid_id(value) then message = tr('Use a unique ID of letters, digits, _ or -.', '请输入由文字、数字、_ 或 - 组成的唯一标识。', '請輸入由文字、數字、_ 或 - 組成的唯一識別碼。'); return render() end
      done(value)
    end)
  end
  local function unused_id(prefix, lookup)
    while #prefix > 112 do prefix = vim.fn.strcharpart(prefix, 0, vim.fn.strchars(prefix) - 1) end
    if not lookup(prefix) then return prefix end
    for index = 2, 10000 do
      local candidate = prefix .. '_' .. index
      if not lookup(candidate) then return candidate end
    end
  end
  local function add_structure()
    if #document.nodes > 0 or #document.edges > 0 or #document.definitions > 0 then
      message = tr('Choose a structure only on an empty graph.', '请在空图中选择基础结构。', '請在空圖中選擇基礎結構。'); return render()
    end
    local revision = generation
    vim.ui.select({tr('Single stream', '单路结构', '單路結構'), tr('Two inputs + add', '双输入相加', '雙輸入相加'),
      tr('Residual scaffold · branch not connected', '残差骨架 · 分支待连接', '殘差骨架 · 分支待連接'),
      tr('Blank canvas', '空白画布', '空白畫布')},
      {prompt = tr('Choose a basic structure; add your own operators', '选择基础结构，再添加自己的算子', '選擇基礎結構，再新增自己的運算子')}, function(_, choice)
        if not choice or not alive() or generation ~= revision or #document.nodes > 0 or #document.edges > 0 or #document.definitions > 0 then return end
        if choice == 4 then
          message = tr('Blank canvas ready. Add inputs, outputs and operators.', '空白画布已就绪，请添加输入、输出和算子。', '空白畫布已就緒，請新增輸入、輸出和運算子。')
          render(); if plane_status then plane_status(message) end
          return
        end
        local nodes = {{id='input', kind='input', inputs={}, outputs={'value'}}}
        if choice == 2 then nodes[#nodes + 1] = {id='input_2', kind='input', inputs={}, outputs={'value'}} end
        if choice ~= 1 then nodes[#nodes + 1] = {id='merge', kind='add', inputs={'left', 'right'}, outputs={'value'}} end
        nodes[#nodes + 1] = {id='output', kind='output', inputs={'value'}, outputs={}}
        local edges
        if choice == 1 then
          edges = {{from={node='input', port='value'}, to={node='output', port='value'}}}
        else
          edges = {{from={node='input', port='value'}, to={node='merge', port='left'}},
            {from={node='merge', port='value'}, to={node='output', port='value'}}}
          if choice == 2 then edges[#edges + 1] = {from={node='input_2', port='value'}, to={node='merge', port='right'}} end
        end
        local operations = {}
        for _, node in ipairs(nodes) do operations[#operations + 1] = {action = 'add_node', node = node} end
        for _, edge in ipairs(edges) do operations[#operations + 1] = {action = 'connect', from = edge.from, to = edge.to} end
        apply_edits(operations, function(candidate)
          candidate.nodes, candidate.edges = vim.deepcopy(nodes), vim.deepcopy(edges)
        end, function()
          selected = 'input'
          message = choice == 3 and tr('Connect your branch to merge.right; the graph is incomplete.', '请将自定义分支连接到 merge.right；当前图尚未完成。', '請將自訂分支連接至 merge.right；目前圖尚未完成。')
            or tr('Insert operators on a connection; runtime remains unverified.', '可在线上插入算子；运行尚未验证。', '可在線上插入運算子；執行尚未驗證。')
          render()
          if plane_status then plane_status(message, selected) end
        end)
      end)
  end
  local function add_node(splice, default_kind)
    local choices = {'input', 'call', 'add', 'concat', 'output'}
    local revision = generation
    local function build_node(kind)
      if not kind or not alive() or generation ~= revision or (splice and not vim.tbl_contains(document.edges, splice)) then return end
      input_id(tr('Unique node ID: ', '唯一节点 ID：', '唯一節點 ID：'), unused_id(kind, find_node), function(id)
        if find_node(id) then message = tr('Node ID already exists.', '节点 ID 已存在。', '節點 ID 已存在。'); return render() end
        local function commit_node(node, definition)
          local operations = {}
          if definition then operations[#operations + 1] = {action = 'add_definition', definition = definition} end
          operations[#operations + 1] = {action = 'add_node', node = node}
          local first, second
          if splice then
            first = {from = vim.deepcopy(splice.from), to = {node = id, port = node.inputs[1]}}
            second = {from = {node = id, port = node.outputs[1]}, to = vim.deepcopy(splice.to)}
            operations[#operations + 1] = {action = 'disconnect', from = splice.from, to = splice.to}
            operations[#operations + 1] = {action = 'connect', from = first.from, to = first.to}
            operations[#operations + 1] = {action = 'connect', from = second.from, to = second.to}
          end
          apply_edits(operations, function(candidate)
            if definition then candidate.definitions[#candidate.definitions + 1] = vim.deepcopy(definition) end
            candidate.nodes[#candidate.nodes + 1] = vim.deepcopy(node)
            if splice then
              for index, edge in ipairs(candidate.edges) do if vim.deep_equal(edge, splice) then table.remove(candidate.edges, index); break end end
              candidate.edges[#candidate.edges + 1] = first
              candidate.edges[#candidate.edges + 1] = second
            end
          end, function()
            selected = id
            if splice then message = tr('Component inserted; validate its contracts.', '组件已插入；请检查其契约。', '元件已插入；請檢查其契約。') end
            render()
            if plane_status then plane_status(message or tr('Node added.', '节点已添加。', '節點已新增。'), id) end
          end)
        end
        local function insert(definition, axis)
          if not alive() or generation ~= revision or (splice and not vim.tbl_contains(document.edges, splice)) then return end
          local node = {id = id, kind = kind, inputs = {}, outputs = {}}
          if kind == 'input' then node.outputs = {'value'}
          elseif kind == 'output' then node.inputs = {'value'}
          elseif kind == 'call' then node.inputs, node.outputs, node.definition, node.call_style = {'value'}, {'value'}, definition, 'positional'
          else node.inputs, node.outputs = {'left', 'right'}, {'value'}; if kind == 'concat' then node.axis = axis end end
          commit_node(node)
        end
        if kind == 'concat' then
          vim.ui.input({prompt = tr('Concatenation axis (integer): ', '拼接轴（整数）：', '串接軸（整數）：')}, function(raw)
            if not raw or not alive() or generation ~= revision then return end
            local axis = raw:match('^%-?%d+$') and tonumber(raw) or nil
            if not axis or axis < -2147483648 or axis > 2147483647 then message = tr('Invalid axis.', '拼接轴无效。', '串接軸無效。'); return render() end
            insert(nil, axis)
          end)
        elseif kind == 'call' then
          local names = {}; for _, definition in ipairs(document.definitions) do names[#names + 1] = definition.id end
          names[#names + 1] = tr('+ New shared definition', '+ 新建共享组件定义', '+ 新增共用元件定義')
          vim.ui.select(names, {prompt = tr('Choose a definition; calls can share it', '选择定义；多个调用可以共享', '選擇定義；多個呼叫可以共用')}, function(_, choice)
            if not choice or not alive() or generation ~= revision then return end
            if choice <= #document.definitions then insert(document.definitions[choice].id); return end
            input_id(tr('Unique definition ID: ', '唯一组件定义 ID：', '唯一元件定義 ID：'), unused_id(id .. '_component', find_definition), function(def_id)
              if find_definition(def_id) then message = tr('Definition ID already exists.', '组件定义 ID 已存在。', '元件定義 ID 已存在。'); return render() end
              local function use_factory(factory, signature)
                if not factory or not alive() or generation ~= revision then return end
                refresh_signatures()
                if #factory > 240 or not factory:match('^[^%s:]+:[^%s:]+$') then message = tr('Invalid factory reference.', '工厂引用无效。', '工廠參照無效。'); return render() end
                if type(signature) == 'table' and (signature.status == 'observed' or signature.status == 'declared') then observed_signatures[factory] = signature end
                local function commit(params)
                  if not alive() or generation ~= revision or (splice and not vim.tbl_contains(document.edges, splice)) then return end
                  local ok, raw = pcall(vim.json.encode, params)
                  if not ok or #raw > 65536 or vim.tbl_count(params) > 64 then
                    message = tr('Parameters exceed the limit.', '参数超过限制。', '參數超過限制。'); return render()
                  end
                  commit_node({id = id, kind = kind, definition = def_id, call_style = 'positional', inputs = {'value'}, outputs = {'value'}},
                    {id = def_id, factory = factory, params = params})
                end
                local function raw_params()
                  vim.ui.input({prompt = tr('Factory parameters (JSON object): ', '工厂参数（JSON 对象）：', '工廠參數（JSON 物件）：'), default = '{}'}, function(raw)
                    if not raw or not alive() or generation ~= revision then return end
                    local ok, params = pcall(vim.json.decode, raw)
                    if #raw > 65536 or not ok or type(params) ~= 'table' or not raw:match('^%s*{') then
                      message = tr('Enter a JSON object up to 64 KiB.', '请输入不超过 64 KiB 的 JSON 对象。', '請輸入不超過 64 KiB 的 JSON 物件。'); return render()
                    end
                    commit(params)
                  end)
                end
                local required, positional_only = {}, false
                for _, param in ipairs(type(signature) == 'table' and type(signature.parameters) == 'table' and signature.parameters or {}) do
                  if type(param) == 'table' and param.required == true and type(param.name) == 'string'
                      and #param.name <= 128 and param.name:match('^[%a_][%w_]*$') then
                    if param.kind == 'POSITIONAL_ONLY' then positional_only = true else required[#required + 1] = param end
                  end
                end
                if positional_only then
                  message = tr('This factory has positional-only constructor parameters; use a wrapper.', '此工厂包含仅位置构造参数，请使用包装工厂。', '此工廠包含僅位置建構參數，請使用包裝工廠。'); return render()
                end
                if #required == 0 then
                  local known = type(signature) == 'table' and (signature.status == 'observed' or signature.status == 'declared')
                    and not signature.truncated and type(signature.parameters) == 'table'
                  return vim.ui.select({known and tr('Add with default parameters', '使用默认参数添加', '使用預設參數新增')
                      or tr('Try without parameters', '尝试不填参数', '嘗試不填參數'),
                    tr('Enter a parameter object', '输入参数对象', '輸入參數物件'), tr('Cancel', '取消', '取消')},
                    {prompt = known and tr('Parameters can also be edited on the block later.', '稍后也可以在模块上编辑参数。', '稍後也可以在模組上編輯參數。')
                      or tr('Signature is unavailable or truncated; required parameters may be missing.', '签名不可用或不完整，可能缺少必填参数。', '簽章不可用或不完整，可能缺少必填參數。')}, function(_, option)
                      if not alive() or generation ~= revision then return end
                      if option == 1 then commit(vim.empty_dict()) elseif option == 2 then raw_params() end
                    end)
                end
                vim.ui.select({tr('Fill required parameters one by one', '逐项填写必填参数', '逐項填寫必填參數'), tr('Enter JSON object', '输入 JSON 对象', '輸入 JSON 物件')},
                  {prompt = tr('Parameters are declarations; runtime remains unverified.', '参数是声明，尚未运行验证。', '參數是宣告，尚未執行驗證。')}, function(_, option)
                    if not option or not alive() or generation ~= revision then return end
                    if option == 2 then return raw_params() end
                    local params = vim.empty_dict()
                    local function ask(index)
                      if not alive() or generation ~= revision then return end
                      if index > #required then return commit(params) end
                      local parameter, name = required[index], required[index].name
                      local types = {tr('Number', '数字', '數字'), tr('Text', '文本', '文字'),
                        tr('On / off', '开 / 关', '開 / 關'), tr('Null', '空值', '空值'),
                        tr('List or object (JSON)', '列表或对象（JSON）', '清單或物件（JSON）'), tr('Cancel', '取消', '取消')}
                      local suggested = ({number = 1, text = 2, boolean = 3, collection = 5})[parameter.type_hint]
                      if suggested then types[suggested] = types[suggested] .. tr(' · signature hint', ' · 签名提示', ' · 簽章提示') end
                      vim.ui.select(types,
                        {prompt = tr('Choose value type for ', '选择参数值类型：', '選擇參數值類型：') .. name}, function(_, kind)
                          if not kind or not alive() or generation ~= revision then return end
                          local function accept(value)
                            if not alive() or generation ~= revision then return end
                            params[name] = value; ask(index + 1)
                          end
                          if kind == 1 or kind == 2 or kind == 5 then
                            local prompt = kind == 1 and tr('Number: ', '数字：', '數字：')
                              or kind == 2 and tr('Text: ', '文本：', '文字：')
                              or tr('JSON list or object: ', 'JSON 列表或对象：', 'JSON 清單或物件：')
                            vim.ui.input({prompt = name .. ' · ' .. prompt}, function(raw)
                              if raw == nil or not alive() or generation ~= revision then return end
                              if #raw > 65536 then message = tr('Value exceeds the limit.', '值超过限制。', '值超過限制。'); return render() end
                              if kind == 2 then return accept(raw) end
                              if kind == 1 then
                                local value = tonumber(raw)
                                if not value or value ~= value or value == math.huge or value == -math.huge then
                                  message = tr('Enter a finite number.', '请输入有限数字。', '請輸入有限數字。'); return render()
                                end
                                return accept(value)
                              end
                              local ok, value = pcall(vim.json.decode, raw)
                              if not ok or type(value) ~= 'table' or not vim.trim(raw):match('^[%[{]') then
                                message = tr('Enter a JSON list or object.', '请输入 JSON 列表或对象。', '請輸入 JSON 清單或物件。'); return render()
                              end
                              accept(value)
                            end)
                          elseif kind == 3 then
                            vim.ui.select({tr('On', '开', '開'), tr('Off', '关', '關'), tr('Cancel', '取消', '取消')},
                              {prompt = name}, function(_, selected_value)
                                if selected_value == 1 then accept(true) elseif selected_value == 2 then accept(false) end
                              end)
                          elseif kind == 4 then accept(vim.NIL) end
                      end)
                    end
                    ask(1)
                  end)
              end
              local sources = {{label = tr('Enter factory reference', '输入工厂引用', '輸入工廠參照'), id = 'manual'}}
              if actions.project_factories then sources[#sources + 1] = {label = tr('Choose a project factory', '选择项目工厂', '選擇專案工廠'), id = 'project'} end
              if actions.registered_factories then sources[#sources + 1] = {label = tr('Choose a registered component', '选择已注册组件', '選擇已註冊元件'), id = 'registered'} end
              if actions.discover_layers then sources[#sources + 1] = {label = tr('Discover installed torch.nn layers', '发现环境中的 torch.nn 层', '探索環境中的 torch.nn 層'), id = 'torch'} end
              if actions.discover_module_operators then sources[#sources + 1] = {label = tr('Discover operators in a Python module', '发现 Python 模块中的算子', '探索 Python 模組中的運算子'), id = 'module'} end
              vim.ui.select(sources, {prompt = tr('Choose a component source', '选择组件来源', '選擇元件來源'),
                format_item = function(item) return item.label end}, function(_, source)
                if not source or not alive() or generation ~= revision then return end
                local source_id = sources[source] and sources[source].id
                if source_id == 'manual' then
                  vim.ui.input({prompt = tr('Factory module:callable: ', '工厂 module:callable：', '工廠 module:callable：')}, use_factory)
                elseif source_id == 'project' or source_id == 'registered' then
                  local lookup = source_id == 'project' and actions.project_factories or actions.registered_factories
                  lookup(function(references)
                    if not alive() or generation ~= revision then return end
                    if #references == 0 then message = tr('No matching factories found.', '没有发现相应工厂。', '沒有發現相應工廠。'); return render() end
                    vim.ui.select(references, {prompt = tr('Candidates · not runtime verified', '候选 · 尚未运行验证', '候選 · 尚未執行驗證'),
                      format_item = function(item) return type(item) == 'table' and item.reference or item end}, function(reference)
                      if type(reference) == 'table' then use_factory(reference.reference, reference.signature)
                      elseif reference then use_factory(reference) end
                    end)
                  end)
                elseif source_id == 'torch' then
                  actions.discover_layers(function(item) if item and item.name then use_factory('torch.nn:' .. item.name, item.signature) end end)
                elseif source_id == 'module' then
                  actions.discover_module_operators(function(item) if item then use_factory(item.reference, item.signature) end end)
                end
              end)
            end)
          end)
        else insert() end
      end)
    end
    if splice or default_kind then build_node(default_kind or 'call')
    else vim.ui.select(choices, {prompt = tr('Choose node type', '选择节点类型', '選擇節點類型')}, build_node) end
  end
  local function ports(node)
    local revision = generation
    local choices = {}
    if node.kind ~= 'input' then choices[#choices + 1] = 'inputs' end
    if node.kind ~= 'output' then choices[#choices + 1] = 'outputs' end
    vim.ui.select(choices, {prompt = tr('Choose port group', '选择端口组', '選擇連接埠群組'),
      format_item = function(group) return group == 'inputs' and tr('Input ports', '输入端口', '輸入連接埠') or tr('Output ports', '输出端口', '輸出連接埠') end}, function(group)
      if not group or not alive() or generation ~= revision then return end
      local list, choose = vim.deepcopy(node[group]), nil
      local function save_ports()
        if not alive() or generation ~= revision or find_node(node.id) ~= node then return end
        local minimum = group == 'inputs' and (node.kind == 'add' or node.kind == 'concat') and 2 or 1
        if #list < minimum or (node.kind == 'output' and #list ~= 1)
            or (vim.tbl_contains({'cast', 'clone', 'mean_dynamic'}, node.kind) and #list ~= 1)
            or (vim.tbl_contains({'assert_flat', 'assert_batch', 'after'}, node.kind) and #list ~= (group == 'inputs' and 2 or 1))
            or ((node.kind == 'add' or node.kind == 'concat') and group == 'outputs' and #list ~= 1) then
          vim.notify(tr('Invalid port count for this node.', '此节点的端口数量无效。', '此節點的連接埠數量無效。'), vim.log.levels.WARN); return choose()
        end
        for _, edge in ipairs(document.edges) do
          local endpoint = group == 'inputs' and edge.to or edge.from
          if endpoint.node == node.id and not vim.tbl_contains(list, endpoint.port) then
            vim.notify(tr('Disconnect edges before removing or renaming their ports.', '请先断开需要删除或重命名端口的连接。', '請先中斷需要刪除或重新命名連接埠的連線。'), vim.log.levels.WARN); return choose()
          end
        end
        if (node.kind == 'input' and group == 'outputs') or (node.kind == 'output' and group == 'inputs') then
          for name in pairs(node.semantics or {}) do
            if not vim.tbl_contains(list, name) then
              vim.notify(tr('Remove the port semantic label before removing or renaming it.', '删除或重命名端口前请先清除其语义标记。', '刪除或重新命名連接埠前請先清除其語意標記。'), vim.log.levels.WARN); return choose()
            end
          end
          for name in pairs(node.draft_rules or {}) do
            if not vim.tbl_contains(list, name) then
              vim.notify(tr('Remove the port draft rules before removing or renaming it.', '删除或重命名端口前请先移除其草稿规则。', '刪除或重新命名連接埠前請先移除其草稿規則。'), vim.log.levels.WARN); return choose()
            end
          end
        end
        local function apply_ports()
          if not alive() or generation ~= revision or find_node(node.id) ~= node then return end
          local changes = {[group] = list}
          for _, key in ipairs(group == 'inputs' and {'input_shapes', 'input_dtypes'} or {'output_shapes', 'output_dtypes'}) do
            if type(node[key]) == 'table' then
              local retained = vim.deepcopy(node[key])
              for name in pairs(retained) do if not vim.tbl_contains(list, name) then retained[name] = nil end end
              changes[key] = next(retained) and retained or vim.NIL
            end
          end
          patch_node(node, changes)
        end
        local order_changes_behavior = node.kind == 'call' and not vim.deep_equal(list, node[group])
          and ((group == 'inputs' and node.call_style == 'positional') or (group == 'outputs' and node.output_style == 'sequence'))
        local lost_shapes, shape_key = {}, group == 'inputs' and 'input_shapes' or 'output_shapes'
        for name in pairs(node[shape_key] or {}) do if not vim.tbl_contains(list, name) then lost_shapes[#lost_shapes + 1] = name end end
        table.sort(lost_shapes)
        local lost_dtypes, dtype_key = {}, group == 'inputs' and 'input_dtypes' or 'output_dtypes'
        for name in pairs(node[dtype_key] or {}) do if not vim.tbl_contains(list, name) then lost_dtypes[#lost_dtypes + 1] = name end end
        table.sort(lost_dtypes)
        local data_may_change = node.kind == 'input' and group == 'outputs' and document.data ~= nil
          and not vim.deep_equal(list, node[group])
        if not order_changes_behavior and #lost_shapes == 0 and #lost_dtypes == 0 and not data_may_change then return apply_ports() end
        local impact = order_changes_behavior and tr('Port order determines positional arguments or tuple/list outputs; this changes how values are mapped.',
          '端口顺序决定位置参数或元组／列表输出的映射，此修改会改变数据对应关系。',
          '連接埠順序決定位置參數或元組／清單輸出的對應，此修改會改變資料對應關係。') or ''
        if #lost_shapes > 0 then impact = impact .. '\n' .. tr('Shape contracts removed: ', '将移除形状契约：', '將移除形狀契約：') .. table.concat(lost_shapes, ', ') end
        if #lost_dtypes > 0 then impact = impact .. '\n' .. tr('Dtype contracts removed: ', '将移除 dtype 契约：', '將移除 dtype 契約：') .. table.concat(lost_dtypes, ', ') end
        if data_may_change then impact = impact .. '\n' .. tr('Review data binding: input names may change and the data declaration is not updated automatically.',
          '请重新检查数据绑定：输入名称可能改变，数据声明不会自动更新。', '請重新檢查資料繫結：輸入名稱可能改變，資料宣告不會自動更新。') end
        vim.ui.select({tr('Cancel', '取消', '取消'), tr('Apply port order', '应用端口顺序', '套用連接埠順序')}, {
          prompt = impact,
        }, function(_, choice) if choice == 2 then apply_ports() elseif alive() and generation == revision then choose() end end)
      end
      choose = function()
        if not alive() or generation ~= revision or find_node(node.id) ~= node then return end
        local entries = {tr('Add port', '添加端口', '新增連接埠')}
        for index, port in ipairs(list) do entries[#entries + 1] = tostring(index) .. ' · ' .. port end
        entries[#entries + 1] = tr('Save ports', '保存端口', '儲存連接埠')
        entries[#entries + 1] = tr('Cancel', '取消', '取消')
        vim.ui.select(entries, {prompt = tr('Ports in order; choose one to edit', '端口按顺序排列；选择一项进行编辑', '連接埠依序排列；選擇一項進行編輯')}, function(_, choice)
          if not choice or not alive() or generation ~= revision then return end
          if choice == 1 then
            if #list >= 64 then vim.notify(tr('Maximum 64 ports.', '最多 64 个端口。', '最多 64 個連接埠。'), vim.log.levels.WARN); return choose() end
            vim.ui.input({prompt = tr('New port name: ', '新端口名称：', '新連接埠名稱：')}, function(raw)
              if not alive() or generation ~= revision then return end
              if raw == nil then return choose() end
              local name = vim.trim(raw)
              if not valid_id(name) or vim.tbl_contains(list, name) then
                vim.notify(tr('Use a unique port name with letters, digits, _ or -.', '请输入由文字、数字、_ 或 - 组成的唯一端口名。', '請輸入由文字、數字、_ 或 - 組成的唯一連接埠名稱。'), vim.log.levels.WARN); return choose()
              end
              list[#list + 1] = name; choose()
            end)
          elseif choice <= #list + 1 then
            local index = choice - 1
            vim.ui.select({tr('Rename', '重命名', '重新命名'), tr('Move earlier', '向前移动', '向前移動'),
              tr('Move later', '向后移动', '向後移動'), tr('Remove', '移除', '移除')}, {prompt = list[index]}, function(_, action)
                if not alive() or generation ~= revision then return end
                if action == 1 then
                  vim.ui.input({prompt = tr('Port name: ', '端口名称：', '連接埠名稱：'), default = list[index]}, function(raw)
                    if not alive() or generation ~= revision then return end
                    if raw == nil then return choose() end
                    local name = vim.trim(raw)
                    if not valid_id(name) or (name ~= list[index] and vim.tbl_contains(list, name)) then
                      vim.notify(tr('Use a unique port name with letters, digits, _ or -.', '请输入由文字、数字、_ 或 - 组成的唯一端口名。', '請輸入由文字、數字、_ 或 - 組成的唯一連接埠名稱。'), vim.log.levels.WARN); return choose()
                    end
                    list[index] = name; choose()
                  end)
                elseif action == 2 and index > 1 then list[index], list[index - 1] = list[index - 1], list[index]; choose()
                elseif action == 3 and index < #list then list[index], list[index + 1] = list[index + 1], list[index]; choose()
                elseif action == 4 then table.remove(list, index); choose()
                else choose() end
              end)
          elseif choice == #list + 2 then save_ports() end
        end)
      end
      choose()
    end)
  end
  local function edit_semantic(node, preferred_port)
    local revision = generation
    local available = node.kind == 'input' and node.outputs or node.inputs
    local function selected(port)
      if not port or not alive() or generation ~= revision then return end
      vim.ui.input({prompt = tr('Semantic label; blank removes it: ', '语义标记；留空可移除：', '語意標記；留空可移除：'),
        default = node.semantics and node.semantics[port] or ''}, function(raw)
        if raw == nil or not alive() or generation ~= revision then return end
        local label = vim.trim(raw)
        if #label > 256 or label:find('[%c]') then
          message = tr('Use at most 256 bytes without control characters.', '请输入不超过 256 字节且不含控制字符的标记。', '請輸入不超過 256 位元組且不含控制字元的標記。'); return render()
        end
        local semantics = vim.deepcopy(node.semantics or vim.empty_dict())
        semantics[port] = label ~= '' and label or nil
        patch_node(node, {semantics = next(semantics) and semantics or vim.NIL}, function()
          message = tr('Label saved in draft; no rule or runtime observation is attached.', '标记已写入草稿；尚未绑定规则或运行观察。', '標記已寫入草稿；尚未繫結規則或執行觀察。'); render()
        end)
      end)
    end
    if preferred_port and vim.tbl_contains(available, preferred_port) then selected(preferred_port)
    else vim.ui.select(available, {prompt = tr('Choose a port to label', '选择要标记的端口', '選擇要標記的連接埠')}, selected) end
  end
  local function edit_draft_rules(node, preferred_port)
    local revision = generation
    local ports = node.kind == 'input' and node.outputs or node.inputs
    local function selected(port)
      if not port or not alive() or generation ~= revision then return end
      local rules = node.draft_rules and node.draft_rules[port] or {}
      local choices = {tr('+ Add draft rule', '+ 添加草稿规则', '+ 新增草稿規則')}
      for _, rule in ipairs(rules) do
        choices[#choices + 1] = rule.kind .. ' · ' .. rule.mode .. (rule.min and (' min=' .. rule.min) or '')
          .. (rule.max and (' max=' .. rule.max) or '') .. (rule.weight and (' weight=' .. rule.weight) or '')
      end
      vim.ui.select(choices, {prompt = tr('Draft only; not enforced. Select a rule to edit or remove.', '仅草稿，尚未执行；选择规则可编辑或移除。', '僅草稿，尚未執行；選擇規則可編輯或移除。')}, function(_, choice)
        if not choice or not alive() or generation ~= revision then return end
        local existing = choice > 1 and vim.deepcopy(rules[choice - 1]) or nil
        local function configure()
        if not existing and #rules >= 32 then message = tr('Rule limit reached.', '规则数已达上限。', '規則數已達上限。'); return render() end
        vim.ui.select({'finite', 'range'}, {prompt = tr('Choose rule kind', '选择规则类型', '選擇規則類型')}, function(kind)
          if not kind or not alive() or generation ~= revision then return end
          local modes = kind == 'range' and {'observe', 'require', 'penalty'} or {'observe', 'require'}
          local mode_labels = {
            observe = tr('observe · record violations', 'observe · 记录违规', 'observe · 記錄違規'),
            require = tr('require · reject violations', 'require · 违规时中止', 'require · 違規時中止'),
            penalty = tr('penalty · add weighted loss', 'penalty · 增加加权损失', 'penalty · 增加加權損失'),
          }
          vim.ui.select(modes, {prompt = tr('Choose action at supported training boundary', '选择受支持训练边界的作用方式', '選擇受支援訓練邊界的作用方式'),
            format_item = function(mode) return mode_labels[mode] or mode end}, function(mode)
            if not mode or not alive() or generation ~= revision then return end
            local rule = {kind = kind, mode = mode}
            local function add()
              if not alive() or generation ~= revision then return end
              local draft = vim.deepcopy(node.draft_rules or vim.empty_dict())
              draft[port] = draft[port] or {}
              if existing then draft[port][choice - 1] = rule
              else draft[port][#draft[port] + 1] = rule end
              patch_node(node, {draft_rules = draft}, function()
                message = tr('Draft changed; existing bound rules are unchanged. Review and bind the new rule before use.',
                  '草稿已修改；已绑定规则不变。使用前请审核并绑定新规则。', '草稿已修改；已繫結規則不變。使用前請檢視並繫結新規則。'); render()
              end)
            end
            local function ask_weight()
              if mode ~= 'penalty' then return add() end
              vim.ui.input({prompt = tr('Positive penalty weight: ', '正数惩罚权重：', '正數懲罰權重：'),
                default = existing and existing.weight and tostring(existing.weight) or ''}, function(raw)
                if raw == nil or not alive() or generation ~= revision then return end
                local value = tonumber(raw)
                if not value or value ~= value or value == math.huge or value <= 0 then message = tr('Weight must be finite and positive.', '权重必须是有限正数。', '權重必須是有限正數。'); return render() end
                rule.weight = value; add()
              end)
            end
            if kind == 'finite' then return add() end
            vim.ui.input({prompt = tr('Minimum (blank for none): ', '最小值（留空表示无）：', '最小值（留空表示無）：'),
              default = existing and existing.min ~= nil and tostring(existing.min) or ''}, function(raw_min)
              if raw_min == nil or not alive() or generation ~= revision then return end
              local min = vim.trim(raw_min) == '' and nil or tonumber(raw_min)
              if vim.trim(raw_min) ~= '' and (not min or min ~= min or math.abs(min) == math.huge) then message = tr('Minimum must be finite.', '最小值必须是有限数。', '最小值必須是有限數。'); return render() end
              vim.ui.input({prompt = tr('Maximum (blank for none): ', '最大值（留空表示无）：', '最大值（留空表示無）：'),
                default = existing and existing.max ~= nil and tostring(existing.max) or ''}, function(raw_max)
                if raw_max == nil or not alive() or generation ~= revision then return end
                local max = vim.trim(raw_max) == '' and nil or tonumber(raw_max)
                if (vim.trim(raw_max) ~= '' and (not max or max ~= max or math.abs(max) == math.huge)) or (not min and not max) or (min and max and min > max) then
                  message = tr('Provide a valid range with at least one bound.', '请提供至少一个有效边界，且最小值不能大于最大值。', '請提供至少一個有效邊界，且最小值不能大於最大值。'); return render()
                end
                rule.min, rule.max = min, max; ask_weight()
              end)
            end)
          end)
        end)
        end
        if not existing then return configure() end
        vim.ui.select({tr('Keep', '保留', '保留'), tr('Edit draft rule', '编辑草稿规则', '編輯草稿規則'),
          tr('Remove draft rule', '移除草稿规则', '移除草稿規則')}, {
          prompt = choices[choice] .. tr(' · bound rules are not changed', ' · 已绑定规则不会改变', ' · 已繫結規則不會變更'),
        }, function(_, action)
          if not alive() or generation ~= revision then return end
          if action == 2 then return configure() end
          if action ~= 3 then return end
          local draft = vim.deepcopy(node.draft_rules)
          table.remove(draft[port], choice - 1)
          if #draft[port] == 0 then draft[port] = nil end
          patch_node(node, {draft_rules = next(draft) and draft or vim.NIL}, function()
            message = tr('Draft removed; bound rules are unchanged.', '草稿已移除；已绑定规则不变。', '草稿已移除；已繫結規則不變。'); render()
          end)
        end)
      end)
    end
    if preferred_port and vim.tbl_contains(ports, preferred_port) then selected(preferred_port)
    else vim.ui.select(ports, {prompt = tr('Choose a port for draft rules', '选择草稿规则端口', '選擇草稿規則連接埠')}, selected) end
  end
  local function port_supervision_hub(node)
    local revision = generation
    local ports = node.kind == 'input' and node.outputs or node.inputs
    local labels = {}
    for _, port in ipairs(ports) do
      local semantic = node.semantics and node.semantics[port]
      local rules = node.draft_rules and node.draft_rules[port] or {}
      labels[#labels + 1] = port .. ' · ' .. (semantic or tr('no semantic label', '未标记语义', '未標記語意'))
        .. ' · ' .. tostring(#rules) .. tr(' draft rules', ' 条草稿规则', ' 條草稿規則')
    end
    vim.ui.select(labels, {prompt = tr('Choose a port to configure supervision', '选择要配置监督的端口', '選擇要設定監督的連接埠')}, function(_, index)
      local port = index and ports[index]
      if not port or not alive() or generation ~= revision then return end
      local choices = {tr('Edit semantic label', '编辑语义标记', '編輯語意標記'),
        tr('Add, edit or remove draft rules', '增删改草稿规则', '增刪改草稿規則'),
        tr('Inspect binding, coverage and violations', '查看绑定、覆盖与违规', '檢視繫結、涵蓋與違規'),
        tr('Run saved model forward', '运行已保存模型的前向', '執行已儲存模型的前向')}
      vim.ui.select(choices, {prompt = labels[index] .. tr(' · drafts are not automatically enabled',
        ' · 草稿不会自动启用', ' · 草稿不會自動啟用')}, function(_, choice)
        if not choice or not alive() or generation ~= revision then return end
        if choice == 1 then edit_semantic(node, port)
        elseif choice == 2 then edit_draft_rules(node, port)
        elseif choice == 3 and actions.supervision then actions.supervision(vim.deepcopy(document), node.id, port)
        elseif choice == 4 and actions.forward then actions.forward(vim.deepcopy(document)) end
      end)
    end)
  end
  local function edit_shape(node)
    local groups = {}
    if #node.inputs > 0 then groups[#groups + 1] = 'inputs' end
    if #node.outputs > 0 and (vim.tbl_contains({'input', 'call'}, node.kind)
        or (actions.merge_output_shapes and vim.tbl_contains({'add', 'concat'}, node.kind))) then groups[#groups + 1] = 'outputs' end
    if #groups == 0 then message = tr('This node has no editable shape contracts.', '此节点没有可编辑的形状契约。', '此節點沒有可編輯的形狀契約。'); return render() end
    local revision = generation
    vim.ui.select(groups, {prompt = tr('Choose port group', '选择端口组', '選擇連接埠群組'),
      format_item = function(group) return group == 'inputs' and tr('Input ports', '输入端口', '輸入連接埠') or tr('Output ports', '输出端口', '輸出連接埠') end}, function(group)
      if not group or not alive() or generation ~= revision then return end
      vim.ui.select(node[group], {prompt = tr('Choose a port', '选择端口', '選擇連接埠')}, function(port)
        if not port or not alive() or generation ~= revision then return end
        local key = group == 'inputs' and 'input_shapes' or 'output_shapes'
        local old = type(node[key]) == 'table' and node[key][port] or nil
        local shape = old and vim.deepcopy(old) or {}
        local choose
        local function valid() return alive() and generation == revision end
        local function save_shape(value)
          if not valid() then return end
          local contracts = vim.deepcopy(node[key] or vim.empty_dict())
          contracts[port] = value
          patch_node(node, {[key] = next(contracts) and contracts or vim.NIL}, function()
            message = tr('Shape is a declaration; runtime is not verified.', '形状只是声明，运行尚未验证。', '形狀只是宣告，執行尚未驗證。'); render()
          end)
        end
        local function numeric(index)
          if not valid() then return end
          local previous = shape[index]
          vim.ui.input({prompt = tr('Positive dimension size: ', '维度大小（正整数）：', '維度大小（正整數）：'),
            default = previous ~= vim.NIL and previous and tostring(previous) or ''}, function(raw)
            if not valid() then return end
            if raw == nil then return choose() end
            local value = raw:match('^%d+$') and tonumber(raw) or nil
            if not value or value < 1 or value > 1000000000000 or value % 1 ~= 0 then
              vim.notify(tr('Enter a positive integer up to 1000000000000.', '请输入不超过 1000000000000 的正整数。', '請輸入不超過 1000000000000 的正整數。'), vim.log.levels.WARN)
              return choose()
            end
            shape[index] = value; choose()
          end)
        end
        choose = function()
          if not valid() then return end
          local values = {}
          for _, dim in ipairs(shape) do values[#values + 1] = dim == vim.NIL and '?' or tostring(dim) end
          local summary = #values == 0 and tr('scalar (zero dimensions)', '标量（零维）', '純量（零維）') or table.concat(values, ' × ')
          if old == nil and #values == 0 then summary = tr('no saved contract; empty draft is scalar', '尚无已保存契约；空草稿表示标量', '尚無已儲存契約；空草稿表示純量') end
          vim.ui.select({tr('Add fixed dimension', '添加固定维度', '新增固定維度'),
            tr('Add dynamic dimension (?)', '添加动态维度（?）', '新增動態維度（?）'),
            tr('Edit or remove a dimension', '修改或移除维度', '修改或移除維度'),
            tr('Save shape', '保存形状', '儲存形狀'), tr('Remove shape contract', '移除形状契约', '移除形狀契約'),
            tr('Cancel', '取消', '取消')}, {prompt = tr('Shape: ', '形状：', '形狀：') .. summary}, function(_, choice)
            if not choice or not valid() then return end
            if choice == 1 then
              if #shape >= 16 then vim.notify(tr('Maximum 16 dimensions.', '最多 16 个维度。', '最多 16 個維度。'), vim.log.levels.WARN); return choose() end
              numeric(#shape + 1)
            elseif choice == 2 then
              if #shape >= 16 then vim.notify(tr('Maximum 16 dimensions.', '最多 16 个维度。', '最多 16 個維度。'), vim.log.levels.WARN); return choose() end
              shape[#shape + 1] = vim.NIL; choose()
            elseif choice == 3 then
              if #shape == 0 then return choose() end
              local labels = {}; for index, value in ipairs(values) do labels[index] = tostring(index) .. ' · ' .. value end
              vim.ui.select(labels, {prompt = tr('Choose dimension', '选择维度', '選擇維度')}, function(_, index)
                if not valid() then return end
                if not index then return choose() end
                vim.ui.select({tr('Set fixed size', '设置固定大小', '設定固定大小'), tr('Set dynamic (?)', '设为动态（?）', '設為動態（?）'),
                  tr('Move earlier', '向前移动', '向前移動'), tr('Move later', '向后移动', '向後移動'), tr('Remove dimension', '移除维度', '移除維度')},
                  {prompt = labels[index]}, function(_, action)
                    if not valid() then return end
                    if action == 1 then return numeric(index)
                    elseif action == 2 then shape[index] = vim.NIL
                    elseif action == 3 and index > 1 then shape[index], shape[index - 1] = shape[index - 1], shape[index]
                    elseif action == 4 and index < #shape then shape[index], shape[index + 1] = shape[index + 1], shape[index]
                    elseif action == 5 then table.remove(shape, index) end
                    choose()
                  end)
              end)
            elseif choice == 4 then save_shape(shape)
            elseif choice == 5 then save_shape(nil) end
          end)
        end
        choose()
      end)
    end)
  end
  local function edit_dtype(node)
    local groups = {}
    if #node.inputs > 0 then groups[#groups + 1] = 'inputs' end
    if #node.outputs > 0 then groups[#groups + 1] = 'outputs' end
    local revision = generation
    vim.ui.select(groups, {prompt = tr('Choose port group', '选择端口组', '選擇連接埠群組'),
      format_item = function(group) return group == 'inputs' and tr('Input ports', '输入端口', '輸入連接埠') or tr('Output ports', '输出端口', '輸出連接埠') end}, function(group)
      if not group or not alive() or generation ~= revision then return end
      vim.ui.select(node[group], {prompt = tr('Choose a port', '选择端口', '選擇連接埠')}, function(port)
        if not port or not alive() or generation ~= revision then return end
        local key = group == 'inputs' and 'input_dtypes' or 'output_dtypes'
        local old = type(node[key]) == 'table' and node[key][port] or ''
        vim.ui.input({prompt = tr('PyTorch dtype name (blank removes contract; resolved on run): ',
          'PyTorch dtype 名称（留空移除契约；运行时解析）：', 'PyTorch dtype 名稱（留空移除契約；執行時解析）：'), default = old}, function(raw)
          if raw == nil or not alive() or generation ~= revision then return end
          local name = vim.trim(raw)
          if name ~= '' and (#name > 64 or not name:match('^[A-Za-z][A-Za-z0-9_]*$')) then
            message = tr('Enter a public dtype name without a module prefix.', '请输入不带模块前缀的公开 dtype 名称。', '請輸入不帶模組前綴的公開 dtype 名稱。'); return render()
          end
          local contracts = vim.deepcopy(node[key] or vim.empty_dict())
          contracts[port] = name ~= '' and name or nil
          patch_node(node, {[key] = next(contracts) and contracts or vim.NIL}, function()
            message = tr('Dtype is declared, not converted or runtime verified.', 'dtype 已声明，不会自动转换，尚未运行验证。', 'dtype 已宣告，不會自動轉換，尚未執行驗證。'); render()
          end)
        end)
      end)
    end)
  end
  local function affected_calls(definition)
    local affected = {[definition.id] = true}
    local changed = true
    while changed do
      changed = false
      for _, parent in ipairs(document.definitions) do
        if not affected[parent.id] then
          for _, child in ipairs(parent.children or {}) do
            if affected[child] then affected[parent.id] = true; changed = true; break end
          end
        end
      end
    end
    local calls = {}
    for _, call in ipairs(document.nodes) do
      local uses = affected[call.definition]
      if not uses then
        for _, id in ipairs(call.dtype_definitions or {}) do if affected[id] then uses = true; break end end
      end
      if uses then calls[#calls + 1] = call.id end
    end
    return calls
  end
  local function confirm_shared_change(definition, revision, apply)
    local calls = affected_calls(definition)
    local preview = {}
    for index = 1, math.min(#calls, 6) do preview[#preview + 1] = calls[index] end
    if #calls > #preview then preview[#preview + 1] = '… +' .. (#calls - #preview) end
    local function choose()
      if not alive() or generation ~= revision or find_definition(definition.id) ~= definition then return end
      vim.ui.select({tr('Cancel', '取消', '取消'), tr('Apply to all shared calls', '应用到全部共享调用', '套用至全部共用呼叫'),
        tr('View affected calls', '查看受影响调用', '查看受影響呼叫')}, {
        prompt = tr('Affected calls: ', '受影响的调用：', '受影響的呼叫：') .. table.concat(preview, ', '),
      }, function(_, choice)
        if not alive() or generation ~= revision or find_definition(definition.id) ~= definition then return end
        if choice == 2 then apply()
        elseif choice == 3 then
          vim.ui.select(calls, {prompt = tr('Affected calls · select one to return', '受影响的调用 · 选择后返回', '受影響的呼叫 · 選擇後返回')}, function() choose() end)
        end
      end)
    end
    choose()
  end
  local function edit_parameter(node, refreshed)
    refresh_signatures()
    local definition = find_definition(node.definition)
    if not definition then return end
    -- A discovery is an observation, not a guarantee: keep its hints available
    -- while editing, label them as earlier evidence, and offer explicit refresh.
    -- The metadata context still invalidates them when the selected environment
    -- or project changes; packages may also change in place between observations.
    local revision, count = generation, 0
    if not refreshed and definition.factory:match('^component:') and actions.registered_factories then
      actions.registered_factories(function(items)
        if not alive() or generation ~= revision or find_definition(node.definition) ~= definition then return end
        observed_signatures[definition.factory] = nil
        for _, item in ipairs(items) do
          if type(item) == 'table' and item.reference == definition.factory then
            observed_signatures[definition.factory] = item.signature; break
          end
        end
        if not observed_signatures[definition.factory] then
          message = tr('Current component metadata is unavailable; manual editing remains available.',
            '当前组件元数据不可用，仍可手动编辑。', '目前元件中繼資料無法使用，仍可手動編輯。')
        end
        edit_parameter(node, true)
      end)
      return
    end
    count = #affected_calls(definition)
    local keys, seen = {}, {}
    for key in pairs(definition.params or {}) do keys[#keys + 1] = key; seen[key] = true end
    table.sort(keys)
    local signature = observed_signatures[definition.factory]
    local required, hints, type_hints, skipped = {}, {}, {}, 0
    if type(signature) == 'table' and type(signature.parameters) == 'table' then
      for _, param in ipairs(signature.parameters) do
        if type(param) == 'table' and type(param.name) == 'string' and param.name:match('^[%a_][%w_]*$') then
          if vim.tbl_contains({'POSITIONAL_ONLY', 'VAR_POSITIONAL', 'VAR_KEYWORD'}, param.kind) then
            skipped = skipped + 1
          else
            if not seen[param.name] then keys[#keys + 1] = param.name; seen[param.name] = true end
            required[param.name] = param.required == true
            type_hints[param.name] = param.type_hint
            if param.default_available == true then
              local ok, raw = pcall(vim.json.encode, param.default)
              if ok and #raw <= 100 then hints[param.name] = raw end
            end
          end
        end
      end
    end
    local labels = {tr('+ Add parameter', '+ 添加参数', '+ 新增參數')}
    for _, key in ipairs(keys) do
      local value = (definition.params or {})[key]
      local ok, encoded = pcall(vim.json.encode, value)
      labels[#labels + 1] = key .. (required[key] and ' *' or '') .. (type_hints[key] and (' [' .. type_hints[key] .. ']') or '') .. ' = '
        .. (value == nil and (hints[key] and (tr('(observed default: ', '（观察到的默认值：', '（觀察到的預設值：') .. hints[key] .. ')')
          or tr('(not set)', '（未设置）', '（未設定）')) or (ok and encoded or '?'))
    end
    local refresh_index
    local module = not definition.factory:match('^component:') and definition.factory:match('^([^:]+):[^:]+$') or nil
    if module and actions.discover_module_operators then
      labels[#labels + 1] = tr('Refresh signature in selected Python', '在所选 Python 中重新检查签名', '在所選 Python 中重新檢查簽章')
      refresh_index = #labels
    end
    local source = signature and signature.status == 'declared' and tr(' · manifest declaration', ' · 清单声明', ' · 清單宣告')
      or signature and signature.status == 'observed' and tr(' · earlier observed signature; refresh after environment changes',
        ' · 先前观察到的签名；环境变化后请重新检查', ' · 先前觀察到的簽章；環境變更後請重新檢查') or ''
    local excluded = skipped > 0 and tr(' · positional/variadic parameters omitted', ' · 已省略仅位置及可变参数', ' · 已省略僅位置及可變參數') or ''
    vim.ui.select(labels, {prompt = tr('Choose a parameter · shared calls: ', '选择参数 · 共享调用数：', '選擇參數 · 共用呼叫數：') .. count .. source .. excluded}, function(_, index)
      if not index or not alive() or generation ~= revision or find_definition(definition.id) ~= definition then return end
      if index == refresh_index then
        local reference = definition.factory
        actions.discover_module_operators(function(item)
          if not alive() or generation ~= revision or find_definition(definition.id) ~= definition
              or definition.factory ~= reference then return end
          refresh_signatures()
          if type(item) ~= 'table' or item.reference ~= reference then
            message = tr('Selected factory differs; existing parameters and signature are unchanged.',
              '所选工厂不同，现有参数和签名未改变。', '所選工廠不同，既有參數和簽章未變更。'); return render()
          end
          observed_signatures[reference] = type(item.signature) == 'table' and item.signature.status == 'observed' and item.signature or nil
          if not observed_signatures[reference] then
            message = tr('The runtime did not report a signature; manual editing remains available.',
              '运行时未报告签名，仍可手动编辑。', '執行階段未回報簽章，仍可手動編輯。')
          end
          edit_parameter(node, true)
        end, module)
        return
      end
      local function set_name(key)
        if not key or not alive() or generation ~= revision then return end
        key = vim.trim(key)
        if #key > 128 or not key:match('^[%a_][%w_]*$') then
          message = tr('Parameter name must be a Python identifier.', '参数名须为 Python 标识符。', '參數名稱須為 Python 識別碼。'); return render()
        end
        local value = (definition.params or {})[key]
        local function apply_value(next_value, completed)
          if not alive() or generation ~= revision or find_definition(definition.id) ~= definition then return end
          if vim.deep_equal(value, next_value) then if completed then completed() end; return end
          local params = vim.deepcopy(definition.params or vim.empty_dict()); params[key] = next_value
          if not next(params) then params = vim.empty_dict() end
          local encoded_ok, result = pcall(vim.json.encode, params)
          if not encoded_ok or #result > 65536 or vim.tbl_count(params) > 64 then
            message = tr('Parameters exceed the limit.', '参数超过限制。', '參數超過限制。'); return render()
          end
          local function apply()
            if not alive() or generation ~= revision or find_definition(definition.id) ~= definition then return end
            local operation = {action = next_value == nil and 'remove_parameter' or 'set_parameter',
              definition = definition.id, name = key}
            if next_value ~= nil then operation.value = next_value end
            apply_edits({operation}, function(candidate)
              for _, item in ipairs(candidate.definitions) do
                if item.id == definition.id then item.params = vim.deepcopy(params); break end
              end
            end, completed)
          end
          if count <= 1 then return apply() end
          confirm_shared_change(definition, revision, apply)
        end
        local options = {tr('Text', '文本', '文字'), tr('Number', '数字', '數字'),
          tr('On / off', '开 / 关', '開 / 關'), tr('Null', '空值', '空值'),
          tr('List or object (JSON)', '列表或对象（JSON）', '清單或物件（JSON）'),
          tr('Remove parameter', '删除参数', '刪除參數'), tr('Cancel', '取消', '取消')}
        local suggested = ({number = 2, text = 1, boolean = 3, collection = 5})[type_hints[key]]
        if suggested then options[suggested] = options[suggested] .. tr(' · signature hint', ' · 签名提示', ' · 簽章提示') end
        vim.ui.select(options, {prompt = tr('Choose value type for ', '选择参数值类型：', '選擇參數值類型：') .. key}, function(_, kind)
          if not kind or not alive() or generation ~= revision then return end
          local function edit_input(initial, title, parse)
            local baseline = vim.deepcopy(document)
            local draft_key = definition.id .. '\0' .. key .. '\0' .. tostring(kind)
            require('model_maven.input_draft').open(input_drafts, draft_key, title, initial, baseline, function(raw, completed)
              if not alive() or generation ~= revision or not vim.deep_equal(document, baseline)
                  or find_definition(definition.id) ~= definition then
                return false, tr('The graph changed; this input remains a draft.', '图已变化；输入仍保留为草稿。', '圖已變更；輸入仍保留為草稿。')
              end
              local valid, parsed = parse(raw)
              if not valid then return false, parsed end
              apply_value(parsed, completed)
              return true
            end, tr)
          end
          if kind == 1 then
            edit_input(type(value) == 'string' and value or '', key, function(raw) return true, raw end)
          elseif kind == 2 then
            edit_input(type(value) == 'number' and tostring(value) or '', key, function(raw)
              local number = tonumber(raw)
              if not number or number ~= number or number == math.huge or number == -math.huge then
                return false, tr('Enter a finite number.', '请输入有限数字。', '請輸入有限數字。')
              end
              return true, number
            end)
          elseif kind == 3 then
            vim.ui.select({tr('On', '开', '開'), tr('Off', '关', '關'), tr('Cancel', '取消', '取消')},
              {prompt = tr('Choose a boolean value', '选择布尔值', '選擇布林值')}, function(_, selected_value)
                if selected_value == 1 then apply_value(true) elseif selected_value == 2 then apply_value(false) end
              end)
          elseif kind == 4 then apply_value(vim.NIL)
          elseif kind == 5 then
            local ok, encoded = pcall(vim.json.encode, value)
            edit_input(value ~= nil and ok and encoded or '{}', key, function(raw)
              local decoded, next_value = pcall(vim.json.decode, raw)
              if not decoded or #raw > 65536 or type(next_value) ~= 'table'
                  or not vim.trim(raw):match('^[%[{]') then
                return false, tr('Enter a JSON list or object.', '请输入 JSON 列表或对象。', '請輸入 JSON 清單或物件。')
              end
              return true, next_value
            end)
          elseif kind == 6 then apply_value(nil) end
        end)
      end
      if index == 1 then vim.ui.input({prompt = tr('Parameter name: ', '参数名：', '參數名稱：')}, set_name)
      else set_name(keys[index - 1]) end
    end)
  end
  local function edit_definition(node)
    local definition = find_definition(node.definition)
    if not definition then message = tr('Definition not found.', '找不到组件定义。', '找不到元件定義。'); return render() end
    local revision = generation
    vim.ui.input({prompt = tr('Factory module:callable: ', '工厂 module:callable：', '工廠 module:callable：'), default = definition.factory}, function(factory)
      if not factory or not alive() or generation ~= revision then return end
      if #factory > 240 or not factory:match('^[^%s:]+:[^%s:]+$') then message = tr('Invalid factory reference.', '工厂引用无效。', '工廠參照無效。'); return render() end
      local ok, prior = pcall(vim.json.encode, definition.params or vim.empty_dict())
      vim.ui.input({prompt = tr('Factory parameters (JSON object): ', '工厂参数（JSON 对象）：', '工廠參數（JSON 物件）：'), default = ok and prior or '{}'}, function(raw)
        if not raw or not alive() or generation ~= revision then return end
        local parsed, params = pcall(vim.json.decode, raw)
        if #raw > 65536 or not parsed or type(params) ~= 'table' or not raw:match('^%s*{') then
          message = tr('Enter a JSON object up to 64 KiB.', '请输入不超过 64 KiB 的 JSON 对象。', '請輸入不超過 64 KiB 的 JSON 物件。'); return render()
        end
        confirm_shared_change(definition, revision, function()
          if not alive() or generation ~= revision or find_definition(definition.id) ~= definition then return end
          apply_edits({{action = 'update_definition', id = definition.id, changes = {factory = factory, params = params}}}, function(candidate)
            for _, item in ipairs(candidate.definitions) do if item.id == definition.id then item.factory, item.params = factory, vim.deepcopy(params); break end end
          end)
        end)
      end)
    end)
  end
  local function replace_factory(node)
    local definition, revision = find_definition(node.definition), generation
    if not definition then return end
    local function current() return alive() and generation == revision and find_definition(definition.id) == definition end
    local function use_factory(factory, signature)
      if not current() or factory == nil then return end
      if type(factory) ~= 'string' or #factory > 240 or not factory:match('^[^%s:]+:[^%s:]+$') then
        message = tr('Enter a valid module:callable factory.', '请输入有效的 module:callable 工厂。', '請輸入有效的 module:callable 工廠。'); return render()
      end
      vim.ui.select({tr('Keep existing parameters', '保留现有参数', '保留現有參數'),
        tr('Clear parameters', '清空参数', '清空參數'), tr('Cancel', '取消', '取消')}, {
        prompt = tr('Replace factory with ', '将工厂替换为 ', '將工廠替換為 ') .. factory
          .. tr(' · existing parameter compatibility is not verified', ' · 现有参数兼容性尚未验证', ' · 現有參數相容性尚未驗證'),
      }, function(_, choice)
        if not current() or (choice ~= 1 and choice ~= 2) then return end
        local params = choice == 1 and vim.deepcopy(definition.params or vim.empty_dict()) or vim.empty_dict()
        confirm_shared_change(definition, revision, function()
          if not current() then return end
          apply_edits({{action = 'update_definition', id = definition.id, changes = {factory = factory, params = params}}}, function(candidate)
            for _, item in ipairs(candidate.definitions) do
              if item.id == definition.id then item.factory, item.params = factory, vim.deepcopy(params); break end
            end
          end, function()
            refresh_signatures()
            observed_signatures[factory] = type(signature) == 'table' and signature or nil
            message = tr('Factory replaced in draft; review parameters and validate.', '工厂已在草稿中替换；请检查参数并验证。', '工廠已在草稿中替換；請檢查參數並驗證。'); render()
          end)
        end)
      end)
    end
    local sources = {{id = 'manual', label = tr('Enter a factory reference', '输入工厂引用', '輸入工廠參照')}}
    if actions.project_factories then sources[#sources + 1] = {id = 'project', label = tr('Choose a project factory', '选择项目工厂', '選擇專案工廠')} end
    if actions.registered_factories then sources[#sources + 1] = {id = 'registered', label = tr('Choose a registered component', '选择已注册组件', '選擇已註冊元件')} end
    if actions.discover_layers then sources[#sources + 1] = {id = 'torch', label = tr('Discover installed torch.nn layers', '发现环境中的 torch.nn 层', '探索環境中的 torch.nn 層')} end
    if actions.discover_module_operators then sources[#sources + 1] = {id = 'module', label = tr('Discover operators in a Python module', '发现 Python 模块中的算子', '探索 Python 模組中的運算子')} end
    vim.ui.select(sources, {prompt = tr('Choose replacement source', '选择替换来源', '選擇替換來源'), format_item = function(item) return item.label end}, function(source)
      if not source or not current() then return end
      if source.id == 'manual' then
        vim.ui.input({prompt = tr('Factory module:callable: ', '工厂 module:callable：', '工廠 module:callable：'), default = definition.factory}, use_factory)
      elseif source.id == 'torch' then
        actions.discover_layers(function(item) if item and item.name then use_factory('torch.nn:' .. item.name, item.signature) end end)
      elseif source.id == 'module' then
        actions.discover_module_operators(function(item) if item then use_factory(item.reference, item.signature) end end)
      else
        local lookup = source.id == 'project' and actions.project_factories or actions.registered_factories
        lookup(function(references)
          if not current() then return end
          if #references == 0 then message = tr('No matching factories found.', '没有发现相应工厂。', '沒有發現相應工廠。'); return render() end
          vim.ui.select(references, {prompt = tr('Candidates · not runtime verified', '候选 · 尚未运行验证', '候選 · 尚未執行驗證'),
            format_item = function(item) return type(item) == 'table' and item.reference or item end}, function(item)
            if type(item) == 'table' then use_factory(item.reference, item.signature) elseif item then use_factory(item) end
          end)
        end)
      end
    end)
  end
  local function edit_container_children(node)
    local definition = find_definition(node.definition)
    if not definition then return end
    local revision, children, style = generation, vim.deepcopy(definition.children or {}), definition.children_style
    local function current() return alive() and generation == revision and find_definition(definition.id) == definition end
    local choose
    choose = function()
      if not current() then return end
      local labels = {tr('+ Add child component', '+ 添加子组件', '+ 新增子元件')}
      for index, id in ipairs(children) do labels[#labels + 1] = tostring(index) .. ' · ' .. id end
      if actions.container_list_children then
        labels[#labels + 1] = tr('Constructor arguments: ', '构造传参方式：', '建構傳參方式：')
          .. (style == 'list' and tr('one ordered list', '一个有序列表', '一個有序清單') or tr('individual positional arguments', '逐个位置参数', '逐個位置參數'))
      end
      labels[#labels + 1] = tr('Save constructor children', '保存构造子组件', '儲存建構子元件')
      labels[#labels + 1] = tr('Cancel', '取消', '取消')
      vim.ui.select(labels, {prompt = tr('Runtime container: choose how children reach the constructor; shared definitions share weights.',
        '运行时容器：选择子组件传入构造函数的方式；共享定义会共享权重。',
        '執行時容器：選擇子元件傳入建構函式的方式；共用定義會共用權重。')}, function(_, choice)
        if not choice or not current() then return end
        if choice == 1 then
          if #children >= 64 then return end
          local options = {}; for _, item in ipairs(document.definitions) do if item.id ~= definition.id then options[#options + 1] = item.id end end
          return vim.ui.select(options, {prompt = tr('Choose a child definition', '选择子组件定义', '選擇子元件定義')}, function(id)
            if id and current() then children[#children + 1] = id; choose() end
          end)
        end
        local style_index = actions.container_list_children and (#children + 2) or nil
        local save_index = #children + (style_index and 3 or 2)
        if choice == style_index then
          return vim.ui.select({tr('Individual positional arguments', '逐个位置参数', '逐個位置參數'),
            tr('One ordered list', '一个有序列表', '一個有序清單')},
            {prompt = tr('Choose how the selected factory receives its children', '选择工厂接收子组件的方式', '選擇工廠接收子元件的方式')}, function(_, selected_style)
              if not selected_style or not current() then return end
              style = selected_style == 2 and 'list' or nil
              choose()
            end)
        end
        if choice == save_index then
          return confirm_shared_change(definition, revision, function()
            if not current() then return end
            local value = #children > 0 and children or vim.NIL
            local changes = {children = value}
            if actions.container_list_children then changes.children_style = style or vim.NIL end
            apply_edits({{action = 'update_definition', id = definition.id, changes = changes}}, function(candidate)
              for _, item in ipairs(candidate.definitions) do
                if item.id == definition.id then item.children = #children > 0 and vim.deepcopy(children) or nil; item.children_style = style; break end
              end
            end)
          end)
        end
        if choice > #children + 1 then return end
        local index = choice - 1
        vim.ui.select({tr('Move earlier', '向前移动', '向前移動'), tr('Move later', '向后移动', '向後移動'), tr('Remove', '移除', '移除')}, {prompt = labels[choice]}, function(_, action)
          if not current() then return end
          if action == 1 and index > 1 then children[index], children[index - 1] = children[index - 1], children[index]
          elseif action == 2 and index < #children then children[index], children[index + 1] = children[index + 1], children[index]
          elseif action == 3 then table.remove(children, index) end
          choose()
        end)
      end)
    end
    choose()
  end
  local function change_definition(node)
    local revision, choices, definitions = generation, {}, {}
    for _, definition in ipairs(document.definitions) do
      definitions[#definitions + 1] = definition
      choices[#choices + 1] = definition.id .. (definition.id == node.definition and ' · current' or '')
    end
    if #choices == 0 then message = tr('No component definitions available.', '没有可用的组件定义。', '沒有可用的元件定義。'); return render() end
    vim.ui.select(choices, {prompt = tr('Select a shared definition for this call', '为此调用选择共享定义', '為此呼叫選擇共用定義')}, function(_, index)
      if not index or not alive() or generation ~= revision or find_node(node.id) ~= node then return end
      local definition = definitions[index]
      if not definition or definition.id == node.definition then return end
      vim.ui.select({tr('Keep current definition', '保留当前定义', '保留目前定義'), tr('Change this call', '更改此调用', '變更此呼叫')}, {
        prompt = tr('Ports and shape declarations stay unchanged and require review.', '端口和形状声明保持不变，需要重新审核。', '連接埠與形狀宣告保持不變，需要重新檢視。'),
      }, function(_, choice)
        if choice ~= 2 or not alive() or generation ~= revision or find_node(node.id) ~= node
            or find_definition(definition.id) ~= definition then return end
        apply_edits({{action = 'update_node', id = node.id, changes = {definition = definition.id}}}, function(candidate)
          for _, item in ipairs(candidate.nodes) do if item.id == node.id then item.definition = definition.id; break end end
        end)
      end)
    end)
  end
  -- Copy the complete constructor dependency graph while preserving sharing
  -- inside the copy. No learned weights are copied.
  local function independent_definitions(roots, requested)
    local used, mapping, order = {}, {}, {}
    for _, item in ipairs(document.definitions) do used[item.id] = true end
    for old, id in pairs(requested or {}) do
      if used[id] then return nil end
      used[id], mapping[old] = true, id
    end
    local visited = {}
    local function visit(id)
      if visited[id] then return true end
      visited[id] = true
      local definition = find_definition(id)
      if not definition then return false end
      if not mapping[id] then
        local fresh = unused_id(id .. '_copy', function(value) return used[value] end)
        if not fresh then return false end
        mapping[id], used[fresh] = fresh, true
      end
      for _, child in ipairs(definition.children or {}) do if not visit(child) then return false end end
      order[#order + 1] = id
      return true
    end
    for _, id in ipairs(roots) do if not visit(id) then return nil end end
    if #document.definitions + #order > 512 then return nil end
    local copies = {}
    for _, id in ipairs(order) do
      local copy = vim.deepcopy(find_definition(id)); copy.id = mapping[id]
      for index, child in ipairs(copy.children or {}) do copy.children[index] = mapping[child] end
      copies[#copies + 1] = copy
    end
    return copies, mapping
  end
  local function make_independent(node)
    local definition, revision = find_definition(node.definition), generation
    if not definition then message = tr('Definition not found.', '找不到组件定义。', '找不到元件定義。'); return render() end
    input_id(tr('New independent definition ID: ', '新的独立组件定义 ID：', '新的獨立元件定義 ID：'), unused_id(definition.id .. '_copy', find_definition), function(id)
      if not alive() or generation ~= revision or find_node(node.id) ~= node or find_definition(definition.id) ~= definition then return end
      if find_definition(id) then message = tr('Definition ID already exists.', '组件定义 ID 已存在。', '元件定義 ID 已存在。'); return render() end
      vim.ui.select({tr('Keep shared instance', '保持共享实例', '保持共用實例'), tr('Create independent instance', '创建独立实例', '建立獨立實例')}, {
        prompt = tr('Uses the same factory and parameters; existing learned weights are not copied.',
          '使用相同工厂和参数，不复制已有训练权重。', '使用相同工廠與參數，不複製既有訓練權重。'),
      }, function(_, choice)
        if choice ~= 2 or not alive() or generation ~= revision or find_node(node.id) ~= node or find_definition(id) then return end
        local copies = independent_definitions({definition.id}, {[definition.id] = id})
        if not copies then message = tr('Cannot copy the full component tree.', '无法复制完整组件树。', '無法複製完整元件樹。'); return render() end
        local operations = {}
        for _, copy in ipairs(copies) do operations[#operations + 1] = {action = 'add_definition', definition = copy} end
        operations[#operations + 1] = {action = 'update_node', id = node.id, changes = {definition = id}}
        apply_edits(operations, function(candidate)
          for _, copy in ipairs(copies) do candidate.definitions[#candidate.definitions + 1] = vim.deepcopy(copy) end
          for _, item in ipairs(candidate.nodes) do if item.id == node.id then item.definition = id; break end end
        end)
      end)
    end)
  end
  local function duplicate_node(node)
    local revision = generation
    input_id(tr('New node ID: ', '新节点 ID：', '新節點 ID：'), unused_id(node.id .. '_copy', find_node), function(id)
      if not alive() or generation ~= revision or find_node(node.id) ~= node then return end
      if find_node(id) then message = tr('Node ID already exists.', '节点 ID 已存在。', '節點 ID 已存在。'); return render() end
      local function insert(new_definitions, new_id)
        if not alive() or generation ~= revision or find_node(node.id) ~= node or find_node(id) then return end
        local operations = {}
        for _, definition in ipairs(new_definitions or {}) do operations[#operations + 1] = {action = 'add_definition', definition = definition} end
        operations[#operations + 1] = {action = 'copy_node', id = node.id, new_id = id}
        if new_id then operations[#operations + 1] = {action = 'update_node', id = id, changes = {definition = new_id}} end
        apply_edits(operations, function(candidate)
          for _, definition in ipairs(new_definitions or {}) do candidate.definitions[#candidate.definitions + 1] = vim.deepcopy(definition) end
          local copy = vim.deepcopy(node); copy.id = id
          if copy.kind == 'output' then copy.output_name = nil end
          if new_id then copy.definition = new_id end
          candidate.nodes[#candidate.nodes + 1] = copy
        end, function()
          selected = id
          message = tr('Node copied without edges; review its ports and connect it.', '节点已复制但未复制连接，请检查端口并接线。', '節點已複製但未複製連線，請檢查連接埠並接線。'); render()
          if plane_status then plane_status(message, id) end
        end)
      end
      if node.kind ~= 'call' then insert(); return end
      local definition = find_definition(node.definition)
      if not definition then message = tr('Definition not found.', '找不到组件定义。', '找不到元件定義。'); return render() end
      vim.ui.select({tr('Share the same component', '共享同一组件', '共用同一元件'),
        tr('Create independent component', '创建独立组件', '建立獨立元件')}, {
        prompt = tr('Copy keeps port contracts but never copies edges or learned weights.',
          '复制保留端口契约，但不复制连接或已训练权重。', '複製保留連接埠契約，但不複製連線或已訓練權重。'),
      }, function(_, choice)
        if not choice or not alive() or generation ~= revision then return end
        if choice == 1 then insert(); return end
        input_id(tr('New independent definition ID: ', '新的独立组件定义 ID：', '新的獨立元件定義 ID：'), unused_id(definition.id .. '_copy', find_definition), function(def_id)
          if not alive() or generation ~= revision or find_definition(def_id) then
            if alive() then message = tr('Definition changed or ID exists.', '组件定义已变化或 ID 已存在。', '元件定義已變更或 ID 已存在。'); render() end
            return
          end
          local copies = independent_definitions({definition.id}, {[definition.id] = def_id})
          if not copies then message = tr('Cannot copy the full component tree.', '无法复制完整组件树。', '無法複製完整元件樹。'); return render() end
          insert(copies, def_id)
        end)
      end)
    end)
  end
  local function connect_start(node)
    if #node.outputs == 0 then message = tr('Select a node with an output port.', '请选择带输出端口的节点。', '請選擇帶輸出連接埠的節點。'); return render() end
    local revision = generation
    vim.ui.select(node.outputs, {prompt = tr('Choose source port', '选择来源端口', '選擇來源連接埠')}, function(port)
      if not port or not alive() or generation ~= revision then return end
      pending = {node = node.id, port = port, generation = generation}
      message = tr('Select a destination node to connect; Esc cancels.', '选择目标节点接线；Esc 取消。', '選擇目標節點接線；Esc 取消。'); render()
    end)
  end
  local function shape_conflict(source, target, input_port)
    local from = find_node(source.node)
    local a = from and type(from.output_shapes) == 'table' and from.output_shapes[source.port] or nil
    local b = type(target.input_shapes) == 'table' and target.input_shapes[input_port] or nil
    if type(a) ~= 'table' or type(b) ~= 'table' then return false end
    if #a ~= #b then return true end
    for i, dim in ipairs(a) do
      if dim ~= vim.NIL and b[i] ~= vim.NIL and dim ~= b[i] then return true end
    end
    return false
  end
  local function connect_ports(source, target_id, port, moving_edge)
    local function notice(text)
      message = text
      render()
      if plane_status then plane_status(text, target_id) end
    end
    local node, from = find_node(target_id), find_node(source.node)
    if not node or not from or generation ~= source.generation
        or (moving_edge and not vim.tbl_contains(document.edges, moving_edge))
        or not vim.tbl_contains(from.outputs, source.port) or not vim.tbl_contains(node.inputs, port) then
      pending = nil; return notice(tr('The graph changed; connection cancelled.', '图已变化，接线已取消。', '圖已變更，接線已取消。'))
    end
    if shape_conflict(source, node, port) then
      return notice(tr('Declared port shapes conflict; edit the contracts before connecting.', '声明的端口形状冲突，请先修改契约。', '宣告的連接埠形狀衝突，請先修改契約。'))
    end
    local existing
    for _, edge in ipairs(document.edges) do
      if edge.to.node == node.id and edge.to.port == port then
        if existing then
          return notice(tr('This input has multiple existing edges; remove duplicates first.', '此输入有多条现有连接，请先移除重复连接。', '此輸入有多條現有連線，請先移除重複連線。'))
        end
        existing = edge
      end
    end
    if existing and (not moving_edge or existing == moving_edge)
        and existing.from.node == source.node and existing.from.port == source.port then
      pending = nil; return notice(tr('These ports are already connected.', '端口已经连接。', '連接埠已連接。'))
    end
    -- Replacing the occupied input removes its old edge before cycle checking.
    local seen, stack, successors = {}, {node.id}, {}
    for _, edge in ipairs(document.edges) do
      if edge ~= existing and edge ~= moving_edge then
        successors[edge.from.node] = successors[edge.from.node] or {}
        successors[edge.from.node][#successors[edge.from.node] + 1] = edge.to.node
      end
    end
    while #stack > 0 do
      local id = table.remove(stack)
      if id == source.node then
        return notice(tr('Connection would create a cycle. Use a component for loops.', '此连接会形成循环，请将循环封装为组件。', '此連線會形成循環，請將循環封裝為元件。'))
      end
      if not seen[id] then
        seen[id] = true
        for _, next_id in ipairs(successors[id] or {}) do stack[#stack + 1] = next_id end
      end
    end
    local function apply()
      if not alive() or generation ~= source.generation or (moving_edge and not vim.tbl_contains(document.edges, moving_edge)) then return end
      local operations, removals = {}, {}
      local function remove(edge)
        if not edge then return end
        removals[#removals + 1] = vim.deepcopy(edge)
        operations[#operations + 1] = {action = 'disconnect', from = vim.deepcopy(edge.from), to = vim.deepcopy(edge.to)}
      end
      remove(existing)
      if moving_edge and moving_edge ~= existing then remove(moving_edge) end
      local edge = {from = {node = source.node, port = source.port}, to = {node = node.id, port = port}}
      operations[#operations + 1] = {action = 'connect', from = vim.deepcopy(edge.from), to = vim.deepcopy(edge.to)}
      apply_edits(operations, function(candidate)
        for index = #candidate.edges, 1, -1 do
          for _, old in ipairs(removals) do
            if vim.deep_equal(candidate.edges[index], old) then table.remove(candidate.edges, index); break end
          end
        end
        candidate.edges[#candidate.edges + 1] = vim.deepcopy(edge)
      end, function(warning)
        notice(warning or tr('Connection updated. Validate to check the full graph.', '连接已更新，请规划检查完整图。', '連線已更新，請規劃檢查完整圖。'))
      end)
    end
    local function review_dtype()
      local source_dtype = type(from.output_dtypes) == 'table' and from.output_dtypes[source.port] or nil
      local target_dtype = type(node.input_dtypes) == 'table' and node.input_dtypes[port] or nil
      if not source_dtype or not target_dtype or source_dtype == target_dtype then return apply() end
      vim.ui.select({tr('Cancel', '取消', '取消'), tr('Connect and review at runtime', '连接并在运行时检查', '連接並於執行時檢查')}, {
        prompt = tr('Declared dtype names differ: ', '声明的 dtype 名称不同：', '宣告的 dtype 名稱不同：')
          .. source_dtype .. ' → ' .. target_dtype
          .. tr('. Names may be aliases; no automatic conversion is applied.',
            '。名称可能是别名；不会自动转换。', '。名稱可能是別名；不會自動轉換。'),
      }, function(_, choice) if choice == 2 then apply() end end)
    end
    if existing and existing ~= moving_edge then
      local edge = existing
      vim.ui.select({tr('Keep existing connection', '保留现有连接', '保留現有連線'), tr('Replace connection', '替换连接', '取代連線')}, {
        prompt = tr('Replace ', '替换 ', '取代 ') .. edge.from.node .. '.' .. edge.from.port .. ' → ' .. node.id .. '.' .. port .. '?',
      }, function(_, choice) if choice == 2 then review_dtype() end end)
    else review_dtype() end
  end
  local function connect_finish(node)
    local source = pending
    if not source or source.generation ~= generation or not find_node(source.node) then pending = nil; return render() end
    if #node.inputs == 0 then message = tr('This node has no input ports.', '此节点没有输入端口。', '此節點沒有輸入連接埠。'); return render() end
    vim.ui.select(node.inputs, {prompt = tr('Choose destination port', '选择目标端口', '選擇目標連接埠')}, function(port)
      if not port or not alive() or not pending or pending ~= source or generation ~= source.generation then return end
      connect_ports(source, node.id, port)
    end)
  end
  local function remove_edge()
    local function disconnect(edge)
      apply_edits({{action = 'disconnect', from = edge.from, to = edge.to}}, function(candidate)
        for index, item in ipairs(candidate.edges) do if vim.deep_equal(item, edge) then table.remove(candidate.edges, index); break end end
      end)
    end
    if selected_edge and vim.tbl_contains(document.edges, selected_edge) then
      local edge, revision = selected_edge, generation
      vim.ui.select({tr('Keep connection', '保留连接', '保留連線'), tr('Remove connection', '移除连接', '移除連線')}, {
        prompt = edge.from.node .. '.' .. edge.from.port .. ' → ' .. edge.to.node .. '.' .. edge.to.port,
      }, function(_, choice)
        if choice ~= 2 or not alive() or generation ~= revision then return end
        disconnect(edge)
      end)
      return
    end
    local labels = {}
    for _, edge in ipairs(document.edges) do labels[#labels + 1] = edge.from.node .. '.' .. edge.from.port .. ' → ' .. edge.to.node .. '.' .. edge.to.port end
    if #labels == 0 then message = tr('No edges.', '没有连接。', '沒有連線。'); return render() end
    local revision = generation
    vim.ui.select(labels, {prompt = tr('Remove connection', '移除连接', '移除連線')}, function(_, index)
      if not index or not alive() or generation ~= revision then return end
      disconnect(document.edges[index])
    end)
  end
  local function edge_details(edge)
    if not edge or not vim.tbl_contains(document.edges, edge) then return end
    local revision = generation
    local source, destination = find_node(edge.from.node), find_node(edge.to.node)
    local function describe_shape(node, field, port)
      local shape = node and node[field] and node[field][port]
      if not shape then return '?' end
      local ok, encoded = pcall(vim.json.encode, shape)
      return ok and encoded or '?'
    end
    local shapes = describe_shape(source, 'output_shapes', edge.from.port) .. ' → '
      .. describe_shape(destination, 'input_shapes', edge.to.port)
    vim.ui.select({tr('Change source', '更换来源', '更換來源'), tr('Change destination', '更换目标', '更換目標'),
      tr('Remove connection', '移除连接', '移除連線'), tr('Insert component here', '在此插入组件', '在此插入元件'),
      tr('Go to source node', '跳转到来源节点', '跳至來源節點'), tr('Go to destination node', '跳转到目标节点', '跳至目標節點')}, {
      prompt = edge.from.node .. '.' .. edge.from.port .. ' → ' .. edge.to.node .. '.' .. edge.to.port
        .. tr(' · declared shapes ', ' · 声明形状 ', ' · 宣告形狀 ') .. shapes,
    }, function(_, choice)
      if not choice or not alive() or generation ~= revision or not vim.tbl_contains(document.edges, edge) then return end
      if choice == 3 then remove_edge(); return end
      if choice == 4 then add_node(edge); return end
      if choice == 5 or choice == 6 then
        local id = choice == 5 and edge.from.node or edge.to.node
        if not find_node(id) then message = tr('That node no longer exists.', '该节点已不存在。', '該節點已不存在。'); return render() end
        selected, selected_edge = id, nil
        if focus_root then focus_root = id end
        message = tr('Selected connected node. Open details to edit.', '已选中连接节点，可打开详情编辑。', '已選取連線節點，可開啟詳情編輯。')
        render()
        if selected_row then
          api.nvim_win_set_cursor(win, {selected_row, 0})
          api.nvim_win_call(win, function() vim.cmd('normal! zz') end)
        end
        return
      end
      local nodes, labels = {}, {}
      for _, node in ipairs(document.nodes) do
        local ports = choice == 1 and node.outputs or node.inputs
        if #ports > 0 then nodes[#nodes + 1] = node; labels[#labels + 1] = node.id end
      end
      if #nodes == 0 then message = tr('No compatible-direction ports are available.', '没有对应方向的可用端口。', '沒有對應方向的可用連接埠。'); return render() end
      vim.ui.select(labels, {prompt = choice == 1 and tr('Choose a new source node', '选择新的来源节点', '選擇新的來源節點')
          or tr('Choose a new destination node', '选择新的目标节点', '選擇新的目標節點')}, function(_, index)
        if not index or not alive() or generation ~= revision then return end
        local node = nodes[index]
        vim.ui.select(choice == 1 and node.outputs or node.inputs, {prompt = tr('Choose port', '选择端口', '選擇連接埠')}, function(port)
          if not port or not alive() or generation ~= revision or not vim.tbl_contains(document.edges, edge) then return end
          if choice == 1 then connect_ports({node = node.id, port = port, generation = generation}, edge.to.node, edge.to.port, edge)
          else connect_ports({node = edge.from.node, port = edge.from.port, generation = generation}, node.id, port, edge) end
        end)
      end)
    end)
  end
  local function navigate_connections(node)
    local revision, candidates, labels = generation, {}, {}
    for _, edge in ipairs(document.edges) do
      local other, label
      if edge.from.node == node.id then
        other = edge.to.node
        label = '→ ' .. edge.from.port .. ' → ' .. other .. '.' .. edge.to.port
      elseif edge.to.node == node.id then
        other = edge.from.node
        label = '← ' .. other .. '.' .. edge.from.port .. ' → ' .. edge.to.port
      end
      if other and find_node(other) then candidates[#candidates + 1], labels[#labels + 1] = other, label end
    end
    if #candidates == 0 then message = tr('This node has no connected neighbours.', '此节点没有已连接的相邻节点。', '此節點沒有已連接的相鄰節點。'); return render() end
    vim.ui.select(labels, {prompt = tr('Choose an upstream or downstream node', '选择上游或下游节点', '選擇上游或下游節點')}, function(_, index)
      local id = index and candidates[index]
      if not id or not alive() or generation ~= revision or not find_node(id) then return end
      selected, selected_edge = id, nil
      if focus_root then focus_root = id end
      message = tr('Connected node selected.', '已选中连接节点。', '已選取連線節點。'); render()
      if selected_row then api.nvim_win_set_cursor(win, {selected_row, 0}); api.nvim_win_call(win, function() vim.cmd('normal! zz') end) end
    end)
  end
  local function remove_node(node)
    local revision, count, losses = generation, 0, 0
    local draft_rule_count = 0; for _, rules in pairs(node.draft_rules or {}) do draft_rule_count = draft_rule_count + #rules end
    local incoming, outgoing = {}, {}
    for _, edge in ipairs(document.edges) do if edge.from.node == node.id or edge.to.node == node.id then count = count + 1 end end
    for _, edge in ipairs(document.edges) do
      if edge.to.node == node.id then incoming[#incoming + 1] = edge end
      if edge.from.node == node.id then outgoing[#outgoing + 1] = edge end
    end
    for _, objective in ipairs(document.objectives or {}) do if objective.output == node.id then losses = losses + 1 end end
    local can_bypass = node.kind == 'call' and #node.inputs == 1 and #node.outputs == 1 and #incoming == 1 and #outgoing > 0
    local upstream = can_bypass and find_node(incoming[1].from.node) or nil
    can_bypass = can_bypass and upstream and upstream.id ~= node.id and vim.tbl_contains(upstream.outputs, incoming[1].from.port)
    local destinations = {}
    if can_bypass then
      for _, edge in ipairs(outgoing) do
        local target = find_node(edge.to.node)
        local key = edge.to.node .. '\0' .. edge.to.port
        if not target or target.id == node.id or target.id == upstream.id or not vim.tbl_contains(target.inputs, edge.to.port)
            or destinations[key] or shape_conflict(incoming[1].from, target, edge.to.port) then can_bypass = false; break end
        destinations[key] = true
        for _, other in ipairs(document.edges) do
          if other ~= edge and other.to.node == edge.to.node and other.to.port == edge.to.port then can_bypass = false; break end
        end
        if not can_bypass then break end
      end
    end
    local choices = {tr('Keep node', '保留节点', '保留節點'), tr('Remove node and attached edges', '移除节点及关联连接', '移除節點及關聯連線')}
    if can_bypass then choices[#choices + 1] = tr('Remove node and reconnect', '移除节点并重连', '移除節點並重新連線') end
    vim.ui.select(choices, {
      prompt = tr('Attached edges / losses / labels / draft rules: ', '关联连接数 / 损失数 / 标记数 / 草稿规则数：', '關聯連線數 / 損失數 / 標記數 / 草稿規則數：')
        .. count .. ' / ' .. losses .. ' / ' .. vim.tbl_count(node.semantics or {}) .. ' / ' .. draft_rule_count,
    }, function(_, choice)
      if (choice ~= 2 and not (choice == 3 and can_bypass)) or not alive() or generation ~= revision then return end
      local operations = {{action = 'remove_node', id = node.id, cascade = true}}
      if choice == 3 then
        for _, edge in ipairs(outgoing) do operations[#operations + 1] = {action = 'connect', from = incoming[1].from, to = edge.to} end
      end
      apply_edits(operations, function(candidate)
        for i = #candidate.edges, 1, -1 do local edge = candidate.edges[i]; if edge.from.node == node.id or edge.to.node == node.id then table.remove(candidate.edges, i) end end
        if choice == 3 then
          for _, edge in ipairs(outgoing) do
            candidate.edges[#candidate.edges + 1] = {from = vim.deepcopy(incoming[1].from), to = vim.deepcopy(edge.to)}
          end
        end
        for i = #(candidate.objectives or {}), 1, -1 do if candidate.objectives[i].output == node.id then table.remove(candidate.objectives, i) end end
        for _, group in ipairs(candidate.groups or {}) do
          for i = #group.nodes, 1, -1 do if group.nodes[i] == node.id then table.remove(group.nodes, i) end end
        end
        for i = #candidate.nodes, 1, -1 do if candidate.nodes[i].id == node.id then table.remove(candidate.nodes, i); break end end
      end, function()
        selected = nil
        if choice == 3 then message = tr('Node removed and connections restored; validate the graph.', '节点已移除并重连；请规划检查图。', '節點已移除並重新連線；請規劃檢查圖。') end
        render()
      end)
    end)
  end
  local function group_node(node)
    local revision, groups = generation, document.groups or {}
    local choices = {tr('Remove from group', '移出分组', '移出分組'), tr('+ New group', '+ 新建分组', '+ 新增分組')}
    for _, group in ipairs(groups) do choices[#choices + 1] = group.id end
    local function assign(target)
      if not alive() or generation ~= revision or find_node(node.id) ~= node then return end
      local was_collapsed = target and collapsed_groups[target]
      local next_groups = vim.deepcopy(document.groups or {})
      for _, group in ipairs(next_groups) do
        for index = #group.nodes, 1, -1 do if group.nodes[index] == node.id then table.remove(group.nodes, index) end end
      end
      if target then
        local group
        for _, item in ipairs(next_groups) do if item.id == target then group = item; break end end
        if not group then group = {id = target, nodes = {}}; next_groups[#next_groups + 1] = group end
        group.nodes[#group.nodes + 1] = node.id
      end
      local groups_value = #next_groups > 0 and next_groups or nil
      local function apply(candidate) candidate.groups = groups_value and vim.deepcopy(groups_value) or nil end
      local function finish()
        if target then collapsed_groups[target] = nil end
        if was_collapsed then layout_changed() end
        render()
      end
      if actions.plan_edits and actions.edit_group_set then
        apply_edits({{action = 'set_groups', groups = groups_value or vim.NIL}}, apply, finish)
      else message = tr('Group planning is unavailable; no change was applied.', '分组规划不可用，未应用修改。', '分組規劃無法使用，未套用修改。'); render() end
    end
    vim.ui.select(choices, {prompt = tr('Choose a display group for ', '为节点选择显示分组：', '為節點選擇顯示分組：') .. node.id}, function(_, choice)
      if not choice or not alive() or generation ~= revision then return end
      if choice == 1 then assign(nil)
      elseif choice == 2 then
        if #groups >= 512 then message = tr('Group limit reached.', '分组数量已达上限。', '分組數量已達上限。'); return render() end
        local function find_group(id) for _, group in ipairs(document.groups or {}) do if group.id == id then return group end end end
        input_id(tr('New group ID: ', '新分组 ID：', '新分組 ID：'), unused_id('group', find_group), function(id)
          if find_group(id) then message = tr('Group already exists.', '分组已存在。', '分組已存在。'); return render() end
          assign(id)
        end)
      else
        local group = groups[choice - 2]
        if group then assign(group.id) end
      end
    end)
  end
  local function duplicate_group(root)
    local revision, group_set, node_set = generation, {[root.id] = true}, {}
    local changed = true
    while changed do
      changed = false
      for _, group in ipairs(document.groups or {}) do
        if group.parent and group_set[group.parent] and not group_set[group.id] then group_set[group.id], changed = true, true end
      end
    end
    local selected_groups, selected_nodes, internal, boundary, objectives = {}, {}, {}, 0, {}
    for _, group in ipairs(document.groups or {}) do
      if group_set[group.id] then
        selected_groups[#selected_groups + 1] = group
        for _, id in ipairs(group.nodes) do
          if node_set[id] or not find_node(id) then message = tr('Group membership is invalid; plan the graph first.', '分组成员无效，请先规划图。', '分組成員無效，請先規劃圖。'); return render() end
          node_set[id] = true
        end
      end
    end
    for _, node in ipairs(document.nodes) do if node_set[node.id] then selected_nodes[#selected_nodes + 1] = node end end
    for _, edge in ipairs(document.edges) do
      if node_set[edge.from.node] and node_set[edge.to.node] then internal[#internal + 1] = edge
      elseif node_set[edge.from.node] or node_set[edge.to.node] then boundary = boundary + 1 end
    end
    for _, objective in ipairs(document.objectives or {}) do if node_set[objective.output] then objectives[#objectives + 1] = objective end end
    if #document.nodes + #selected_nodes > 512 or #(document.groups or {}) + #selected_groups > 512
        or #document.edges + #internal > 4096 or #(document.objectives or {}) + #objectives > 64 then
      message = tr('Copy would exceed graph limits.', '复制后将超过图的容量限制。', '複製後將超過圖的容量限制。'); return render()
    end
    vim.ui.select({tr('Share component instances', '共享组件实例', '共用元件實例'), tr('Create independent component instances', '创建独立组件实例', '建立獨立元件實例')}, {
      prompt = tr('Copies internal edges only; external edges omitted: ', '只复制内部连接；省略外部连接：', '只複製內部連線；省略外部連線：') .. boundary
        .. tr(' · learned weights are not copied', ' · 不复制已训练权重', ' · 不複製已訓練權重'),
    }, function(_, choice)
      if not choice or not alive() or generation ~= revision then return end
      local candidate, operations = vim.deepcopy(document), {}
      local node_names, group_names, def_names = {}, {}, {}
      for _, node in ipairs(candidate.nodes) do node_names[node.id] = true end
      for _, group in ipairs(candidate.groups or {}) do group_names[group.id] = true end
      for _, definition in ipairs(candidate.definitions) do def_names[definition.id] = true end
      local function new_name(old, used)
        local id = unused_id(old .. '_copy', function(value) return used[value] end)
        if id then used[id] = true end
        return id
      end
      local node_map, group_map, def_map = {}, {}, {}
      for _, node in ipairs(selected_nodes) do node_map[node.id] = new_name(node.id, node_names) end
      for _, group in ipairs(selected_groups) do group_map[group.id] = new_name(group.id, group_names) end
      if choice == 2 then
        local roots = {}
        for _, node in ipairs(selected_nodes) do
          for _, id in ipairs(node.dtype_definitions or {}) do roots[#roots + 1] = id end
          if node.definition then roots[#roots + 1] = node.definition end
        end
        local copies, mapping = independent_definitions(roots)
        if not copies then message = tr('Cannot copy the full component tree.', '无法复制完整组件树。', '無法複製完整元件樹。'); return render() end
        def_map = mapping
        for _, copy in ipairs(copies) do
          candidate.definitions[#candidate.definitions + 1] = copy
          operations[#operations + 1] = {action = 'add_definition', definition = copy}
        end
      end
      if #candidate.definitions > 512 then message = tr('Copy would exceed the component limit.', '复制后组件数量将超限。', '複製後元件數量將超限。'); return render() end
      for _, node in ipairs(selected_nodes) do
        if not node_map[node.id] then message = tr('Cannot allocate a node ID.', '无法分配节点 ID。', '無法配置節點 ID。'); return render() end
        local copy = vim.deepcopy(node); copy.id = node_map[node.id]
        if copy.kind == 'output' then copy.output_name = nil end
        if copy.definition and def_map[copy.definition] then copy.definition = def_map[copy.definition] end
        for index, reference in ipairs(copy.dtype_definitions or {}) do copy.dtype_definitions[index] = def_map[reference] or reference end
        candidate.nodes[#candidate.nodes + 1] = copy
        operations[#operations + 1] = {action = 'add_node', node = copy}
      end
      for _, group in ipairs(selected_groups) do
        if not group_map[group.id] then message = tr('Cannot allocate a group ID.', '无法分配分组 ID。', '無法配置分組 ID。'); return render() end
        local copy = vim.deepcopy(group); copy.id = group_map[group.id]
        copy.parent = group.parent and (group_map[group.parent] or group.parent) or nil
        for index, id in ipairs(copy.nodes) do copy.nodes[index] = node_map[id] end
        candidate.groups[#candidate.groups + 1] = copy
        operations[#operations + 1] = {action = 'add_group', group = copy}
      end
      for _, edge in ipairs(internal) do
        local copy = vim.deepcopy(edge); copy.from.node, copy.to.node = node_map[edge.from.node], node_map[edge.to.node]
        candidate.edges[#candidate.edges + 1] = copy
        operations[#operations + 1] = {action = 'connect', from = copy.from, to = copy.to}
      end
      for _, objective in ipairs(objectives) do
        local copy = vim.deepcopy(objective); copy.output = node_map[objective.output]
        candidate.objectives[#candidate.objectives + 1] = copy
        operations[#operations + 1] = {action = 'add_objective', objective = copy}
      end
      local ok, raw = pcall(vim.json.encode, candidate)
      if not ok or #raw > 128 * 1024 then message = tr('Copied graph exceeds the save limit.', '复制后的图超过保存限制。', '複製後的圖超過儲存限制。'); return render() end
      local function completed()
        selected = selected_nodes[1] and node_map[selected_nodes[1].id] or nil
        message = tr('Group copied; connect its external ports and review data bindings.', '分组已复制；请连接外部端口并检查数据绑定。', '分組已複製；請連接外部連接埠並檢查資料繫結。'); render()
      end
      if actions.plan_edits and actions.edit_group_copy then
        apply_edits(operations, function(draft)
          for key in pairs(draft) do draft[key] = nil end
          for key, value in pairs(vim.deepcopy(candidate)) do draft[key] = value end
        end, completed)
      else
        message = tr('This CLI cannot safely plan group copies; select a newer CLI.',
          '此 CLI 无法安全规划分组复制，请选择较新版本。', '此 CLI 無法安全規劃分組複製，請選擇較新版本。'); render()
      end
    end)
  end
  local function manage_groups()
    local groups, revision = document.groups or {}, generation
    if #groups == 0 then message = tr('Select a node to create a group.', '请选择节点创建分组。', '請選擇節點建立分組。'); return render() end
    vim.ui.select(groups, {prompt = tr('Choose a group', '选择分组', '選擇分組'),
      format_item = function(group) return group.id .. ' · ' .. #group.nodes end}, function(group)
      if not group or not alive() or generation ~= revision then return end
      vim.ui.select({tr('Toggle collapse', '切换折叠', '切換摺疊'), tr('Move group', '移动分组', '移動分組'),
        tr('Remove group, keep nodes and child groups', '删除分组，保留节点及子分组', '刪除分組，保留節點及子分組'),
        tr('Duplicate group and contents', '复制分组及内容', '複製分組及內容')},
        {prompt = group.id}, function(_, choice)
          if not choice or not alive() or generation ~= revision then return end
          if choice == 1 then
            collapsed_groups[group.id] = not collapsed_groups[group.id] or nil
            if collapsed_groups[group.id] then selected, selected_edge = nil, nil end
            layout_changed(); render()
          elseif choice == 2 then
            local by_id = {}; for _, item in ipairs(groups) do by_id[item.id] = item end
            local labels, parents = {tr('Top level', '顶层', '頂層')}, {false}
            for _, item in ipairs(groups) do
              local current, seen, invalid = item.id, {}, false
              while current do
                if current == group.id or seen[current] or not by_id[current] then invalid = true; break end
                seen[current] = true; current = by_id[current].parent
              end
              if not invalid then labels[#labels + 1] = item.id; parents[#parents + 1] = item.id end
            end
            vim.ui.select(labels, {prompt = tr('Move under group', '移动到父分组', '移動至父分組')}, function(_, index)
              if not index or not alive() or generation ~= revision then return end
              local candidate = vim.deepcopy(document.groups)
              for _, item in ipairs(candidate) do if item.id == group.id then item.parent = parents[index] or nil; break end end
              if actions.plan_edits and actions.edit_group_set then
                apply_edits({{action = 'set_groups', groups = candidate}}, function(draft) draft.groups = vim.deepcopy(candidate) end)
              else message = tr('Group planning is unavailable; no change was applied.', '分组规划不可用，未应用修改。', '分組規劃無法使用，未套用修改。'); render() end
            end)
          elseif choice == 3 then
            local candidate = vim.deepcopy(document.groups)
            for _, child in ipairs(candidate) do if child.parent == group.id then child.parent = group.parent end end
            for index, item in ipairs(candidate) do if item.id == group.id then table.remove(candidate, index); break end end
            local groups_value = #candidate > 0 and candidate or nil
            local function apply(draft) draft.groups = groups_value and vim.deepcopy(groups_value) or nil end
            local function finish() collapsed_groups[group.id] = nil; layout_changed(); render() end
            if actions.plan_edits and actions.edit_group_set then
              apply_edits({{action = 'set_groups', groups = groups_value or vim.NIL}}, apply, finish)
            else message = tr('Group planning is unavailable; no change was applied.', '分组规划不可用，未应用修改。', '分組規劃無法使用，未套用修改。'); render() end
          elseif choice == 4 then
            duplicate_group(group)
          end
        end)
    end)
  end
  local function edit_objective(node)
    local revision, choices, indices = generation, {tr('+ Add loss', '+ 添加损失', '+ 新增損失')}, {}
    for index, objective in ipairs(document.objectives or {}) do
      if objective.output == node.id then
        indices[#indices + 1] = index
        choices[#choices + 1] = objective.loss .. ' → ' .. objective.target .. ' × ' .. tostring(objective.weight)
      end
    end
    vim.ui.select(choices, {prompt = tr('Losses for ', '输出损失：', '輸出損失：') .. node.id}, function(_, choice)
      if not choice or not alive() or generation ~= revision or find_node(node.id) ~= node then return end
      local index = choice > 1 and indices[choice - 1] or nil
      local existing = index and document.objectives[index] or nil
      local function choose_target(done)
        local data = document.data
        local params = type(data) == 'table' and type(data.params) == 'table' and data.params or {}
        local candidates, seen = {}, {}
        local function add(name)
          if valid_id(name) and not seen[name] then seen[name] = true; candidates[#candidates + 1] = name end
        end
        if data and data.factory == 'model_maven_worker.data.presets:graph_image_batch' then
          add(params.target_name)
        elseif data and data.factory == 'model_csv:batch' and type(params.targets) == 'table' then
          for name in pairs(params.targets) do if name ~= '$tensor' then add(name) end end
        end
        table.sort(candidates)
        local default = existing and existing.target or node.id
        local function manual() input_id(tr('Target name in the batch: ', '批次中的目标名称：', '批次中的目標名稱：'), default, done) end
        if #candidates == 0 then return manual() end
        local labels = vim.deepcopy(candidates)
        labels[#labels + 1] = tr('Enter another target name', '输入其他目标名称', '輸入其他目標名稱')
        vim.ui.select(labels, {prompt = tr('Choose a declared data target', '选择已声明的数据目标', '選擇已宣告的資料目標')}, function(_, selected)
          if not selected or not alive() or generation ~= revision then return end
          if selected > #candidates then manual() else done(candidates[selected]) end
        end)
      end
      local function configure()
        vim.ui.select({tr('Mean squared error', '均方误差', '均方誤差'), tr('Cross entropy', '交叉熵', '交叉熵'), tr('Custom loss', '自定义损失', '自訂損失')},
          {prompt = tr('Choose a loss', '选择损失', '選擇損失')}, function(_, loss_choice)
            if not loss_choice or not alive() or generation ~= revision then return end
            local function set_loss(loss)
              if not loss or not alive() or generation ~= revision then return end
              if #loss > 240 or not (loss == 'mse' or loss == 'cross_entropy' or loss:match('^[^%s:]+:[^%s:]+$')) then
                message = tr('Invalid loss reference.', '损失引用无效。', '損失參照無效。'); return render()
              end
              choose_target(function(target)
                vim.ui.input({prompt = tr('Loss weight: ', '损失权重：', '損失權重：'), default = tostring(existing and existing.weight or 1)}, function(raw)
                  if not raw or not alive() or generation ~= revision then return end
                  local weight = tonumber(raw)
                  if not weight or weight ~= weight or weight == math.huge or weight == -math.huge or weight < 0 then
                    message = tr('Weight must be finite and nonnegative.', '权重必须是有限非负数。', '權重必須是有限非負數。'); return render()
                  end
                  if not index and #(document.objectives or {}) >= 64 then message = tr('At most 64 losses.', '最多 64 个损失。', '最多 64 個損失。'); return render() end
                  local objectives = vim.deepcopy(document.objectives or {})
                  local value = {output=node.id, target=target, loss=loss, weight=weight}
                  if index then objectives[index] = value else objectives[#objectives + 1] = value end
                  patch_graph({objectives = objectives})
                end)
              end)
            end
            if loss_choice == 1 then set_loss('mse') elseif loss_choice == 2 then set_loss('cross_entropy')
            else vim.ui.input({prompt = tr('Loss module:callable: ', '损失 module:callable：', '損失 module:callable：'), default = existing and existing.loss or ''}, set_loss) end
          end)
      end
      if not index then return configure() end
      vim.ui.select({tr('Edit loss', '编辑损失', '編輯損失'), tr('Remove loss', '删除损失', '刪除損失')},
        {prompt = choices[choice]}, function(_, action)
          if not action or not alive() or generation ~= revision then return end
          if action == 1 then configure() else
            local objectives = vim.deepcopy(document.objectives)
            table.remove(objectives, index)
            patch_graph({objectives = #objectives > 0 and objectives or vim.NIL})
          end
        end)
    end)
  end
  local function edit_batch_factory()
    local revision, previous = generation, vim.deepcopy(document.data)
    vim.ui.input({prompt = tr('Batch factory module:callable: ', '批次工厂 module:callable：', '批次工廠 module:callable：'),
      default = previous and previous.factory or ''}, function(factory)
      if not factory or not alive() or generation ~= revision then return end
      if #factory > 240 or not factory:match('^[^%s:]+:[^%s:]+$') then
        message = tr('Invalid batch factory.', '批次工厂无效。', '批次工廠無效。'); return render()
      end
      local bindings = {'auto', 'envelope', 'single', 'single-raw', 'args', 'kwargs'}
      vim.ui.select(bindings, {prompt = tr('Batch binding (auto infers it at runtime)', '批次绑定（auto 在运行时推断）', '批次繫結（auto 在執行時推斷）')}, function(binding)
        if not binding or not alive() or generation ~= revision then return end
        local params = previous and previous.factory == factory and previous.params or vim.empty_dict()
        local ok, raw_default = pcall(vim.json.encode, params)
        vim.ui.input({prompt = tr('Batch parameters as JSON object: ', '批次参数 JSON 对象：', '批次參數 JSON 物件：'),
          default = ok and raw_default or '{}'}, function(raw)
          if not raw or not alive() or generation ~= revision then return end
          local parsed, value = pcall(vim.json.decode, raw)
          if #raw > 16384 or not parsed or type(value) ~= 'table' or not raw:match('^%s*{') or vim.tbl_count(value) > 64 then
            message = tr('Enter a JSON object up to 16 KiB with at most 64 parameters.', '请输入不超过 16 KiB、最多 64 个参数的 JSON 对象。', '請輸入不超過 16 KiB、最多 64 個參數的 JSON 物件。'); return render()
          end
          for name in pairs(value) do
            if type(name) ~= 'string' or #name > 128 or not name:match('^[%a_][%w_]*$') then
              message = tr('Batch parameter names must be Python identifiers.', '批次参数名须为 Python 标识符。', '批次參數名稱須為 Python 識別碼。'); return render()
            end
          end
          patch_graph({data = {factory = factory, binding = binding ~= 'auto' and binding or nil, params = value}}, function()
            message = tr('Batch declared; the factory has not run.', '批次已声明，工厂尚未运行。', '批次已宣告，工廠尚未執行。'); render()
          end)
        end)
      end)
    end)
  end
  local function edit_image_transforms()
    if not actions.image_transform_catalog or not document.data or type(document.data.params) ~= 'table' then return end
    local snapshot, revision = vim.deepcopy(document), generation
    local params = document.data.params
    local function current() return alive() and generation == revision and vim.deep_equal(document, snapshot) end
    actions.image_transform_catalog(snapshot, function(catalog)
      if not current() then return end
      local by_kind, kinds = {}, {}
      for _, item in ipairs(catalog) do
        if type(item) == 'table' and type(item.kind) == 'string' and item.kind:match('^[%w_]+$')
            and #item.kind <= 64 and type(item.parameters) == 'table' and vim.islist(item.parameters)
            and #item.parameters <= 64 and not by_kind[item.kind] then
          by_kind[item.kind] = item; kinds[#kinds + 1] = item.kind
        end
      end
      if #kinds == 0 then message = tr('No transform metadata available.', '没有可用的变换元数据。', '沒有可用的變換中繼資料。'); return render() end
      vim.ui.select({'train', 'validation', 'test'}, {prompt = tr('Choose split to edit', '选择要编辑的切分', '選擇要編輯的切分')}, function(split)
        if not split or not current() then return end
        if (params.split_transforms ~= nil and (type(params.split_transforms) ~= 'table' or (next(params.split_transforms) and vim.islist(params.split_transforms))))
            or (params.split_transforms ~= nil and params.transforms ~= nil)
            or (params.split_transforms == nil and params.transforms == nil) then
          message = tr('Configure an explicit image pipeline before editing per split.',
            '请先配置明确的图像流水线，再按切分编辑。', '請先設定明確的影像流水線，再依切分編輯。'); return render()
        end
        local all = vim.deepcopy(params.split_transforms or {})
        if params.transforms ~= nil then
          if type(params.transforms) ~= 'table' or not vim.islist(params.transforms) then
            message = tr('The existing transform declaration is unsupported.', '现有变换声明不受支持。', '現有變換宣告不受支援。'); return render()
          end
          for _, name in ipairs({'train', 'validation', 'test'}) do all[name] = vim.deepcopy(params.transforms) end
        end
        local steps = vim.deepcopy(all[split] or {})
        if type(steps) ~= 'table' or not vim.islist(steps) or #steps > 64 then
          message = tr('This split has an unsupported pipeline.', '此切分的流水线不受支持。', '此切分的流水線不受支援。'); return render()
        end
        local choose
        local function edit_fields(step, adding)
          local schema = by_kind[step.type]
          if not schema then message = tr('Unknown step; use the source editor to preserve it.', '未知步骤；请使用源码编辑器保留它。', '未知步驟；請使用原始碼編輯器保留它。'); return render() end
          local candidate = vim.deepcopy(step)
          local function finish()
            if not current() then return end
            if adding then steps[#steps + 1] = candidate end
            choose()
          end
          local function ask_fields(index)
            if not current() then return end
            local field = schema.parameters[index]
            if not field then
              if schema.custom_parameters then
                local ok, encoded = pcall(vim.json.encode, candidate.parameters or vim.empty_dict())
                return vim.ui.input({prompt = tr('Custom transform parameters (JSON object): ', '自定义变换参数（JSON 对象）：', '自訂變換參數（JSON 物件）：'),
                  default = ok and encoded or '{}'}, function(raw)
                  if not raw or not current() then return end
                  local valid, value = pcall(vim.json.decode, raw)
                  if #raw > 16384 or not valid or type(value) ~= 'table' or not raw:match('^%s*{') then
                    message = tr('Enter a JSON object up to 16 KiB.', '请输入不超过 16 KiB 的 JSON 对象。', '請輸入不超過 16 KiB 的 JSON 物件。'); return render()
                  end
                  candidate.parameters = next(value) and value or vim.empty_dict()
                  if not adding then for key in pairs(step) do step[key] = nil end; for key, value in pairs(candidate) do step[key] = value end end
                  finish()
                end)
              end
              if not adding then for key in pairs(step) do step[key] = nil end; for key, value in pairs(candidate) do step[key] = value end end
              return finish()
            end
            if type(field) ~= 'table' or type(field.name) ~= 'string' or not field.name:match('^[%a_][%w_]*$') then
              message = tr('Unsupported transform metadata.', '变换元数据不受支持。', '變換中繼資料不受支援。'); return render()
            end
            local value = candidate[field.name]
            local ok, encoded = pcall(vim.json.encode, value)
            vim.ui.input({prompt = field.name .. ' · ' .. tostring(field.example or '') .. tr(' (JSON; empty removes): ', '（JSON；留空移除）：', '（JSON；留空移除）：'),
              default = value ~= nil and ok and encoded or ''}, function(raw)
              if raw == nil or not current() then return end
              if vim.trim(raw) == '' then
                if field.required then message = tr('This parameter is required.', '此参数为必填项。', '此參數為必填項。'); return render() end
                candidate[field.name] = nil
              else
                local parsed, decoded = pcall(vim.json.decode, raw)
                if not parsed or #raw > 16384 then message = tr('Enter a valid JSON value.', '请输入有效 JSON 值。', '請輸入有效 JSON 值。'); return render() end
                candidate[field.name] = decoded
              end
              ask_fields(index + 1)
            end)
          end
          if schema.requires_entrypoint then
            return vim.ui.input({prompt = tr('Transform factory module:callable: ', '变换工厂 module:callable：', '變換工廠 module:callable：'),
              default = candidate.entrypoint or ''}, function(reference)
              if not reference or not current() then return end
              if #reference > 240 or not reference:match('^[^%s:]+:[^%s:]+$') then
                message = tr('Enter a module:callable reference.', '请输入 module:callable 引用。', '請輸入 module:callable 參照。'); return render()
              end
              candidate.entrypoint = reference; ask_fields(1)
            end)
          end
          ask_fields(1)
        end
        choose = function()
          if not current() then return end
          local labels = {tr('+ Add transform', '+ 添加变换', '+ 新增變換')}
          for index, step in ipairs(steps) do labels[#labels + 1] = tostring(index) .. ' · ' .. tostring(type(step) == 'table' and step.type or '?'):gsub('[%c]', ' ') end
          labels[#labels + 1] = tr('Save ordered pipeline to graph draft', '将有序流水线保存到图草稿', '將有序流水線儲存至圖草稿')
          labels[#labels + 1] = tr('Cancel', '取消', '取消')
          vim.ui.select(labels, {prompt = split .. tr(' · order matters; not yet run', ' · 顺序影响结果；尚未运行', ' · 順序影響結果；尚未執行')}, function(_, choice)
            if not choice or not current() then return end
            if choice == 1 then
              if #steps >= 64 then message = tr('Maximum 64 steps.', '最多 64 个步骤。', '最多 64 個步驟。'); return render() end
              return vim.ui.select(kinds, {prompt = tr('Choose transform type', '选择变换类型', '選擇變換類型')}, function(kind)
                if kind and current() then edit_fields({type = kind}, true) end
              end)
            end
            if choice == #steps + 2 then
              all[split] = steps
              local active = params.split or 'train'
              if type(all[active]) ~= 'table' or not vim.islist(all[active]) then
                message = tr('Configure the active split pipeline first.', '请先配置当前使用切分的流水线。', '請先設定目前使用切分的流水線。'); return render()
              end
              local ok, raw = pcall(vim.json.encode, all)
              if not ok or #raw > 65536 then message = tr('Pipeline is too large.', '流水线过大。', '流水線過大。'); return render() end
              local data = vim.deepcopy(document.data)
              data.params.transforms = nil; data.params.split_transforms = all
              return patch_graph({data = data}, function()
                message = tr('Pipeline saved in graph draft; preview before running.', '流水线已保存到图草稿；运行前请预览。', '流水線已儲存至圖草稿；執行前請預覽。'); render()
              end)
            end
            if choice > #steps + 1 then return end
            local index, step = choice - 1, steps[choice - 1]
            vim.ui.select({tr('Edit parameters', '编辑参数', '編輯參數'), tr('Move earlier', '向前移动', '向前移動'),
              tr('Move later', '向后移动', '向後移動'), tr('Remove', '移除', '移除')}, {prompt = labels[choice]}, function(_, action)
              if not current() then return end
              if action == 1 then edit_fields(step, false)
              elseif action == 2 and index > 1 then steps[index], steps[index - 1] = steps[index - 1], steps[index]; choose()
              elseif action == 3 and index < #steps then steps[index], steps[index + 1] = steps[index + 1], steps[index]; choose()
              elseif action == 4 then table.remove(steps, index); choose()
              else choose() end
            end)
          end)
        end
        choose()
      end)
    end)
  end
  local function edit_csv_split()
    local revision, data = generation, vim.deepcopy(document.data)
    if not data or data.factory ~= 'model_csv:batch' or type(data.params) ~= 'table' then return end
    local function current() return alive() and generation == revision end
    local options = {tr('Use all rows', '使用全部行', '使用全部列'), 'train', 'val', 'test'}
    vim.ui.select(options, {prompt = tr('CSV split (verify that your adapter supports these arguments)',
      'CSV 切分（请确认当前适配器支持这些参数）', 'CSV 切分（請確認目前配接器支援這些參數）')}, function(_, choice)
      if not choice or not current() then return end
      if choice == 1 then
        data.params.split, data.params.split_weights, data.params.split_seed = nil, nil, nil
        return patch_graph({data = data}, function()
          message = tr('Split removed in draft; preview before running.', '已从草稿移除切分；运行前请预览。', '已從草稿移除切分；執行前請預覽。'); render()
        end)
      end
      local names, weights = {'train', 'val', 'test'}, {}
      local old = type(data.params.split_weights) == 'table' and data.params.split_weights or {}
      local function ask(index)
        if not current() then return end
        if index > #names then
          if weights.train + weights.val + weights.test == 0 then
            message = tr('At least one split weight must be positive.', '至少一个切分权重必须为正。', '至少一個切分權重必須為正。'); return render()
          end
          local default = type(data.params.split_seed) == 'number' and tostring(data.params.split_seed) or ''
          return vim.ui.input({prompt = tr('Non-negative split seed: ', '非负切分种子：', '非負切分種子：'), default = default}, function(raw)
            if raw == nil or not current() then return end
            local seed = raw:match('^%d+$') and tonumber(raw) or nil
            if not seed or seed > 9007199254740991 then
              message = tr('Enter an exactly representable non-negative integer.', '请输入可精确表示的非负整数。', '請輸入可精確表示的非負整數。'); return render()
            end
            data.params.split, data.params.split_weights, data.params.split_seed = options[choice], weights, seed
            patch_graph({data = data}, function()
              message = tr('Split saved in draft; preview the actual adapter before running.', '切分已保存到草稿；运行前请预览实际适配器。', '切分已儲存至草稿；執行前請預覽實際配接器。'); render()
            end)
          end)
        end
        local name = names[index]
        vim.ui.input({prompt = name .. tr(' weight (integer 0–1000000): ', ' 权重（0–1000000 的整数）：', ' 權重（0–1000000 的整數）：'),
          default = type(old[name]) == 'number' and tostring(old[name]) or ''}, function(raw)
          if raw == nil or not current() then return end
          local value = raw:match('^%d+$') and tonumber(raw) or nil
          if not value or value > 1000000 then message = tr('Invalid split weight.', '切分权重无效。', '切分權重無效。'); return render() end
          weights[name] = value; ask(index + 1)
        end)
      end
      ask(1)
    end)
  end
  local function edit_batch_pipeline()
    local revision, original = generation, vim.deepcopy(document.data)
    if not original or type(original.factory) ~= 'string' then return end
    local wrapper = 'model_maven_worker.data.pipeline:batch'
    local config
    if original.factory == wrapper then
      config = vim.deepcopy(original.params)
      if type(config) ~= 'table' or type(config.factory) ~= 'string' or type(config.factory_params) ~= 'table'
          or type(config.split_transforms) ~= 'table' or type(config.split) ~= 'string' then
        message = tr('This pipeline declaration cannot be edited here.', '此流水线声明无法在此编辑。', '此流水線宣告無法在此編輯。'); return render()
      end
    else
      config = {factory = original.factory, factory_params = vim.deepcopy(original.params or vim.empty_dict()),
        split_transforms = vim.empty_dict()}
    end
    local function current() return alive() and generation == revision end
    local choices = {'train', 'validation', 'test'}
    for name in pairs(config.split_transforms) do if type(name) == 'string' and not vim.tbl_contains(choices, name) then choices[#choices + 1] = name end end
    local unwrap_index
    if original.factory == wrapper then
      unwrap_index = #choices + 1
      choices[unwrap_index] = tr('Remove wrapper and retain source factory', '移除包装器并保留源工厂', '移除包裝器並保留來源工廠')
    end
    choices[#choices + 1] = tr('Enter another split', '输入其他切分', '輸入其他切分')
    vim.ui.select(choices, {prompt = tr('Choose a transform pipeline; source factory split parameters remain unchanged.',
      '选择变换流水线；源工厂的切分参数不会自动更改。', '選擇變換流水線；來源工廠的切分參數不會自動變更。')}, function(_, choice)
      if not choice or not current() then return end
      if choice == unwrap_index then
        return vim.ui.select({tr('Cancel', '取消', '取消'), tr('Remove wrapper', '移除包装器', '移除包裝器')}, {
          prompt = tr('All split transform declarations will be removed; the source factory and its parameters remain.',
            '将移除全部切分变换声明；源工厂及其参数保留。', '將移除全部切分變換宣告；來源工廠及其參數保留。')}, function(_, selected)
          if selected ~= 2 or not current() then return end
          patch_graph({data = {factory = config.factory, params = vim.deepcopy(config.factory_params), binding = original.binding}})
        end)
      end
      local function select_split(split)
        if not current() or not split or split == '' or #split > 128 or split:find('[%c]') then return end
        vim.ui.input({prompt = tr('Transform seed (same seed on each call): ', '变换种子（每次调用使用同一值）：', '變換種子（每次呼叫使用相同值）：'),
          default = type(config.seed) == 'number' and tostring(config.seed) or ''}, function(raw)
          if raw == nil or not current() then return end
          local seed = raw:match('^%d+$') and tonumber(raw) or nil
          if not seed or seed > 9007199254740991 then message = tr('Enter an exactly representable non-negative seed.', '请输入可精确表示的非负种子。', '請輸入可精確表示的非負種子。'); return render() end
          local steps = vim.deepcopy(config.split_transforms[split] or {})
          if not vim.islist(steps) or #steps > 64 then return end
          local choose
          local function edit_custom(position)
            local old = position and steps[position] or nil
            if old and (type(old) ~= 'table' or old.type ~= 'custom') then
              message = tr('This step must be edited in its source declaration.', '此步骤请在源码声明中编辑。', '此步驟請在原始碼宣告中編輯。'); return render()
            end
            vim.ui.input({prompt = tr('Target batch path (e.g. kwargs.features): ', '批次目标路径（如 kwargs.features）：', '批次目標路徑（如 kwargs.features）：'),
              default = old and old.target or ''}, function(target)
              if not target or target == '' or #target > 512 or target:find('[%c]') or not current() then return end
              vim.ui.input({prompt = tr('Transform module:callable: ', '变换 module:callable：', '變換 module:callable：'),
                default = old and old.entrypoint or ''}, function(reference)
                if not reference or #reference > 240 or not reference:match('^[^%s:]+:[^%s:]+$') or not current() then return end
                local ok, encoded = pcall(vim.json.encode, old and old.parameters or vim.empty_dict())
                vim.ui.input({prompt = tr('Optional keyword parameters as JSON object: ', '可选关键字参数 JSON 对象：', '可選關鍵字參數 JSON 物件：'), default = ok and encoded or '{}'}, function(raw_params)
                  if not raw_params or #raw_params > 65536 or not current() then return end
                  local valid, params = pcall(vim.json.decode, raw_params)
                  if not valid or type(params) ~= 'table' or not raw_params:match('^%s*{') then message = tr('Parameters must be an object.', '参数必须是对象。', '參數必須是物件。'); return render() end
                  local value = old and vim.deepcopy(old) or {type = 'custom'}
                  if not old then
                    local ids = {}; for _, item in ipairs(steps) do ids[item.id] = true end
                    value.id = unused_id('transform', function(id) return ids[id] end)
                    if not value.id then return end
                  end
                  value.target, value.entrypoint, value.parameters = target, reference, params
                  if position then steps[position] = value else steps[#steps + 1] = value end
                  choose()
                end)
              end)
            end)
          end
          choose = function()
            if not current() then return end
            local labels = {tr('+ Add custom transform', '+ 添加自定义变换', '+ 新增自訂變換')}
            for index, step in ipairs(steps) do labels[#labels + 1] = tostring(index) .. ' · ' .. tostring(step.id or '?') .. ' · ' .. tostring(step.target or '?') end
            labels[#labels + 1] = tr('Save this pipeline', '保存此流水线', '儲存此流水線')
            labels[#labels + 1] = tr('Cancel', '取消', '取消')
            vim.ui.select(labels, {prompt = split .. tr(' · ordered steps; project augmentation can also apply separately',
              ' · 有序步骤；项目级增强也可能另行应用', ' · 有序步驟；專案級增強也可能另行套用')}, function(_, index)
              if not index or not current() then return end
              if index == 1 then
                if #steps >= 64 then return end
                return edit_custom(nil)
              end
              if index == #steps + 2 then
                config.split, config.seed = split, seed
                config.split_transforms[split] = steps
                local ok, raw = pcall(vim.json.encode, config)
                if not ok or #raw > 65536 then message = tr('Pipeline is too large.', '流水线过大。', '流水線過大。'); return render() end
                return patch_graph({data = {factory = wrapper, params = config, binding = original.binding}}, function()
                  message = tr('Pipeline saved in draft; source must return a batch. Preview before running.',
                    '流水线已存入草稿；源工厂必须返回批次。运行前请预览。', '流水線已存入草稿；來源工廠必須回傳批次。執行前請預覽。'); render()
                end)
              end
              if index > #steps + 1 then return end
              local position = index - 1
              vim.ui.select({tr('Edit transform', '编辑变换', '編輯變換'), tr('Move earlier', '向前移动', '向前移動'), tr('Move later', '向后移动', '向後移動'), tr('Remove', '移除', '移除')}, {prompt = labels[index]}, function(_, action)
                if not current() then return end
                if action == 1 then return edit_custom(position)
                elseif action == 2 and position > 1 then steps[position], steps[position - 1] = steps[position - 1], steps[position]
                elseif action == 3 and position < #steps then steps[position], steps[position + 1] = steps[position + 1], steps[position]
                elseif action == 4 then table.remove(steps, position) end
                choose()
              end)
            end)
          end
          choose()
        end)
      end
      if choice == #choices then vim.ui.input({prompt = tr('Split name: ', '切分名称：', '切分名稱：')}, select_split)
      else select_split(choices[choice]) end
    end)
  end
  local function details(requested_action)
    if selected_edge then edge_details(selected_edge); return end
    local node = selected and find_node(selected)
    if not node then message = tr('Select a node first.', '请先选择节点。', '請先選擇節點。'); return render() end
    local entries = {
      {tr('Move to group', '移动到分组', '移動至分組'), 'group'},
      {tr('Edit ports', '编辑端口', '編輯連接埠'), 'ports'},
      {tr('Edit declared shape', '编辑声明形状', '編輯宣告形狀'), 'shape'},
      {tr('Connect from this node', '从此节点接线', '從此節點接線'), 'connect'},
      {tr('Go to a connected node', '跳转到连接节点', '跳至連線節點'), 'neighbours'},
      {tr('Duplicate node', '复制节点', '複製節點'), 'duplicate'},
      {tr('Remove connection', '移除连接', '移除連線'), 'remove_edge'},
      {tr('Remove node', '移除节点', '移除節點'), 'remove_node'},
    }
    if actions.port_dtypes then entries[#entries + 1] = {tr('Edit declared dtype', '编辑声明 dtype', '編輯宣告 dtype'), 'dtype'} end
    if actions.open_generated_node then entries[#entries + 1] = {tr('Locate generated forward code', '定位生成的前向代码', '定位產生的前向程式碼'), 'generated_source'} end
    if node.kind == 'call' then
      local shared_count = 0
      for _, call in ipairs(document.nodes) do if call.definition == node.definition then shared_count = shared_count + 1 end end
      if shared_count > 1 then
        entries[#entries + 1] = {tr('Find calls sharing this component', '查看共享此组件的调用', '查看共用此元件的呼叫') .. ' · ' .. shared_count, 'shared_calls'}
      end
      entries[#entries + 1] = {tr('Call inputs: ', '输入传参方式：', '輸入傳參方式：') .. (node.call_style or 'keyword'), 'call_style'}
      if actions.call_output_style then
        entries[#entries + 1] = {tr('Call outputs: ', '输出映射方式：', '輸出對應方式：') .. (node.output_style or tr('default', '默认', '預設')), 'output_style'}
      end
      entries[#entries + 1] = {tr('Edit one component parameter', '逐项编辑组件参数', '逐項編輯元件參數'), 'parameter'}
      entries[#entries + 1] = {tr('Open component source', '打开组件源码', '開啟元件原始碼'), 'source'}
      entries[#entries + 1] = {tr('Edit shared definition', '编辑共享组件定义', '編輯共用元件定義'), 'edit_definition'}
      entries[#entries + 1] = {tr('Replace component factory', '替换组件工厂', '替換元件工廠'), 'replace_factory'}
      if actions.runtime_containers then entries[#entries + 1] = {tr('Edit runtime container children', '编辑运行时容器子组件', '編輯執行時容器子元件'), 'container_children'} end
      entries[#entries + 1] = {tr('Choose another definition', '选择其他组件定义', '選擇其他元件定義'), 'change_definition'}
      entries[#entries + 1] = {tr('Make this call independent', '让此调用独立', '讓此呼叫獨立'), 'independent'}
    elseif node.kind == 'output' then
      if actions.output_name then entries[#entries + 1] = {tr('Edit external output name', '编辑对外输出名称', '編輯對外輸出名稱'), 'output_name'} end
      entries[#entries + 1] = {tr('Configure losses and targets', '配置损失与目标', '設定損失與目標'), 'objective'}
      entries[#entries + 1] = {tr('Run saved model forward', '运行已保存模型的前向', '執行已儲存模型的前向'), 'forward'}
      entries[#entries + 1] = {tr('Inspect observed supervision', '查看已观察的监督', '檢視已觀察的監督'), 'supervision'}
    elseif node.kind == 'concat' then
      entries[#entries + 1] = {tr('Edit concatenation axis', '编辑拼接轴', '編輯串接軸'), 'axis'}
    elseif node.kind == 'mean_dynamic' then
      entries[#entries + 1] = {tr('Choose dynamic axes', '选择动态轴', '選擇動態軸'), 'dynamic_axes'}
    elseif node.kind == 'assert_flat' then
      entries[#entries + 1] = {tr('Edit expected width', '编辑预期宽度', '編輯預期寬度'), 'flat_width'}
    elseif node.kind == 'cast' and actions.factory_contract then
      entries[#entries + 1] = {tr('Choose dtype sources in order', '按顺序选择类型来源', '依順序選擇型別來源'), 'cast_sources'}
    end
    if node.kind == 'input' or node.kind == 'output' then
      entries[#entries + 1] = {tr('Port supervision', '端口监督', '連接埠監督'), 'port_supervision'}
      entries[#entries + 1] = {tr('Label port semantics', '标记端口语义', '標記連接埠語意'), 'semantic'}
      entries[#entries + 1] = {tr('Edit draft rules', '编辑草稿规则', '編輯草稿規則'), 'draft_rules'}
      entries[#entries + 1] = {tr('Connect data', '连接数据', '連接資料'), 'data'}
      entries[#entries + 1] = {tr('Use a batch factory', '使用批次工厂', '使用批次工廠'), 'batch_factory'}
      if actions.generic_batch_transforms and document.data then
        entries[#entries + 1] = {tr('Edit generic batch transforms by split', '按切分编辑通用批次变换', '依切分編輯通用批次變換'), 'batch_pipeline'}
      end
      if document.data and document.data.factory == 'model_csv:batch' then
        entries[#entries + 1] = {tr('Configure CSV split', '配置 CSV 切分', '設定 CSV 切分'), 'csv_split'}
      end
      if document.data and document.data.factory == 'model_maven_worker.data.presets:graph_image_batch'
          and type(document.data.params) == 'table' and type(document.data.params.split_ratios) == 'table' then
        entries[#entries + 1] = {tr('Choose image data split', '切换图像数据切分', '切換影像資料切分'), 'image_split'}
      end
      if actions.image_transform_catalog and document.data
          and document.data.factory == 'model_maven_worker.data.presets:graph_image_batch' then
        entries[#entries + 1] = {tr('Edit image transforms by split', '按切分编辑图像变换', '依切分編輯影像變換'), 'image_transforms'}
      end
      entries[#entries + 1] = {tr('Preview declared batch', '预览声明的批次', '預覽宣告的批次'), 'preview_data'}
      if node.kind == 'input' then entries[#entries + 1] = {tr('Inspect observed supervision', '查看已观察的监督', '檢視已觀察的監督'), 'supervision'} end
    end
    local priority = {parameter = 1, data = 1, objective = 1, axis = 1, dynamic_axes = 1, flat_width = 1,
      cast_sources = 1, port_supervision = 1, semantic = 2, draft_rules = 3, preview_data = 4, shape = 5, dtype = 6, ports = 7}
    for index, entry in ipairs(entries) do entry[3] = index end
    table.sort(entries, function(a, b)
      local left, right = priority[a[2]] or 10, priority[b[2]] or 10
      return left == right and a[3] < b[3] or left < right
    end)
    local choices = {}; for _, entry in ipairs(entries) do choices[#choices + 1] = entry[1] end
    local revision = generation
    local function dispatch(_, choice)
      if (not choice and not requested_action) or not alive() or generation ~= revision or find_node(node.id) ~= node then return end
      local action = requested_action or (entries[choice] and entries[choice][2])
      local allowed = false
      for _, entry in ipairs(entries) do if entry[2] == action then allowed = true; break end end
      if not allowed then return end
      if action == 'shared_calls' then
        local calls = {}
        for _, call in ipairs(document.nodes) do if call.definition == node.definition then calls[#calls + 1] = call.id end end
        vim.ui.select(calls, {prompt = tr('Select a call to locate', '选择要定位的调用', '選擇要定位的呼叫')}, function(id)
          if not id or not alive() or generation ~= revision or not find_node(id) then return end
          selected, selected_edge = id, nil
          render()
          if plane_status then plane_status(tr('Selected shared call.', '已选择共享调用。', '已選取共用呼叫。'), id) end
        end)
      elseif action == 'ports' then ports(node) elseif action == 'shape' then edit_shape(node)
      elseif action == 'dtype' then edit_dtype(node)
      elseif action == 'port_supervision' then port_supervision_hub(node)
      elseif action == 'semantic' then edit_semantic(node)
      elseif action == 'image_transforms' then edit_image_transforms()
      elseif action == 'draft_rules' then edit_draft_rules(node)
      elseif action == 'csv_split' then edit_csv_split()
      elseif action == 'batch_pipeline' then edit_batch_pipeline()
      elseif action == 'group' then group_node(node)
      elseif action == 'connect' then connect_start(node) elseif action == 'duplicate' then duplicate_node(node)
      elseif action == 'neighbours' then navigate_connections(node)
      elseif action == 'remove_edge' then remove_edge() elseif action == 'remove_node' then remove_node(node)
      elseif action == 'edit_definition' then edit_definition(node) elseif action == 'change_definition' then change_definition(node)
      elseif action == 'replace_factory' then replace_factory(node)
      elseif action == 'container_children' then edit_container_children(node)
      elseif action == 'parameter' then edit_parameter(node)
      elseif action == 'generated_source' and actions.open_generated_node then
        local snapshot = vim.deepcopy(document)
        actions.open_generated_node(snapshot, node.id, function(source_right)
          if alive() and generation == revision and vim.deep_equal(document, snapshot) then show_source_side(source_right) end
        end)
      elseif action == 'source' and actions.open_component_source then
        local definition = find_definition(node.definition)
        if definition then
          local snapshot = vim.deepcopy(document)
          actions.open_component_source(snapshot, definition.factory, function(source_right)
            if alive() and generation == revision and vim.deep_equal(document, snapshot) then show_source_side(source_right) end
          end)
        end
      elseif action == 'objective' then edit_objective(node)
      elseif action == 'batch_factory' then edit_batch_factory()
      elseif action == 'image_split' then
        local params, names = document.data.params, {}
        for _, name in ipairs({'train', 'validation', 'test'}) do
          if type(params.split_ratios[name]) == 'number' and params.split_ratios[name] > 0 then names[#names + 1] = name end
        end
        if #names == 0 then message = tr('No split has a positive weight.', '没有正权重的切分。', '沒有正權重的切分。'); return render() end
        vim.ui.select(names, {prompt = tr('Select split; save and apply data before running', '选择切分；运行前保存并应用数据', '選擇切分；執行前儲存並套用資料')}, function(split)
          if not split or not alive() or generation ~= revision or find_node(node.id) ~= node then return end
          local data = vim.deepcopy(document.data); data.params.split = split
          patch_graph({data = data}, function()
            message = tr('Split changed in draft; preview it before running.', '切分已在草稿中更改；运行前请预览。', '切分已在草稿中變更；執行前請預覽。'); render()
          end)
        end)
      elseif action == 'forward' and actions.forward then actions.forward(vim.deepcopy(document))
      elseif action == 'supervision' and actions.supervision then
        local snapshot = vim.deepcopy(document)
        if node.kind == 'input' and #node.outputs > 1 then
          vim.ui.select(node.outputs, {prompt = tr('Select the observed input port', '选择要查看的输入端口', '選擇要檢視的輸入連接埠')}, function(port)
            if port and alive() and generation == revision then actions.supervision(snapshot, node.id, port) end
          end)
        else actions.supervision(snapshot, node.id, node.kind == 'input' and node.outputs[1] or nil) end
      elseif action == 'preview_data' and actions.preview_data then actions.preview_data(vim.deepcopy(document))
      elseif action == 'data' and actions.configure_data then
        local snapshot = vim.deepcopy(document)
        local function configure(port)
          if not alive() or generation ~= revision then return end
          actions.configure_data(snapshot, node.id, port, function(data)
            if not data or not alive() or generation ~= revision or not vim.deep_equal(document, snapshot) then return end
            patch_graph({data = data}, function()
              message = tr('Data declared; save the graph and configure the project before running.', '数据已声明；运行前请保存图并配置项目。', '資料已宣告；執行前請儲存圖並設定專案。'); render()
              if actions.preview_data then
                local declared, current_generation = vim.deepcopy(document), generation
                vim.ui.select({tr('Preview real batch', '预览真实批次', '預覽真實批次'), tr('Continue editing', '继续编辑', '繼續編輯')}, {
                  prompt = tr('Preview executes the selected data factory; model compatibility remains unverified.',
                    '预览会执行所选数据工厂；模型兼容性仍未验证。', '預覽會執行所選資料工廠；模型相容性仍未驗證。'),
                }, function(_, choice)
                  if choice == 1 and alive() and generation == current_generation and vim.deep_equal(document, declared) then
                    actions.preview_data(declared)
                  end
                end)
              end
            end)
          end)
        end
        if node.kind == 'input' and #node.outputs > 1 then
          vim.ui.select(node.outputs, {prompt = tr('Select the input port to map', '选择要映射的输入端口', '選擇要對應的輸入連接埠')}, function(port)
            if port then configure(port) end
          end)
        else configure(nil) end
      elseif action == 'independent' then make_independent(node)
      elseif action == 'call_style' then
        local styles = {tr('Positional · in port order', '位置参数 · 按端口顺序', '位置參數 · 按連接埠順序'),
          tr('Keyword · named by port', '关键字参数 · 使用端口名称', '關鍵字參數 · 使用連接埠名稱')}
        if actions.list_call then styles[#styles + 1] = tr('One list · in port order', '单个列表 · 按端口顺序', '單個清單 · 按連接埠順序') end
        vim.ui.select(styles,
          {prompt = tr('How should this component receive inputs?', '组件如何接收输入？', '元件如何接收輸入？')}, function(_, index)
            if not index or not alive() or generation ~= revision or find_node(node.id) ~= node then return end
            patch_node(node, {call_style = index == 1 and 'positional' or index == 2 and 'keyword' or 'list'})
          end)
      elseif action == 'output_style' then
        vim.ui.select({tr('Default · one value or named mapping', '默认 · 单值或具名字典', '預設 · 單值或具名字典'),
          tr('Named mapping · keys match output ports', '具名字典 · 键对应输出端口', '具名字典 · 鍵對應輸出連接埠'),
          tr('Tuple or list · in output port order', '元组或列表 · 按输出端口顺序', '元組或清單 · 依輸出連接埠順序')}, {
            prompt = tr('How does this component return outputs? Runtime behavior is not yet verified.',
              '组件如何返回输出？运行行为尚未验证。', '元件如何回傳輸出？執行行為尚未驗證。')}, function(_, index)
            if not index or not alive() or generation ~= revision or find_node(node.id) ~= node then return end
            patch_node(node, {output_style = index == 1 and vim.NIL or index == 2 and 'mapping' or 'sequence'})
          end)
      elseif action == 'cast_sources' then
        local sources = vim.deepcopy(node.dtype_definitions or (node.definition and {node.definition}) or {})
        local tracking = node.track_sequence == true
        local function choose_sources()
          if not alive() or generation ~= revision or find_node(node.id) ~= node then return end
          local labels = {}
          for _, definition in ipairs(document.definitions) do
            local position
            for i, id in ipairs(sources) do if id == definition.id then position = i end end
            labels[#labels + 1] = (position and (tostring(position) .. '. ') or '+ ') .. definition.id .. ' · ' .. definition.factory
          end
          if actions.dtype_sequence_guard then
            labels[#labels + 1] = tracking and tr('Sequence tracking: on', '顺序跟踪：开', '順序追蹤：開')
              or tr('Sequence tracking: off', '顺序跟踪：关', '順序追蹤：關')
          end
          local apply_index = #labels + 1
          labels[#labels + 1] = tr('Apply', '应用', '套用')
          labels[#labels + 1] = tr('Cancel', '取消', '取消')
          vim.ui.select(labels, {prompt = tr('Toggle definitions in parameter search order; no selection uses the default dtype.',
            '按参数搜索顺序选择组件；不选则使用默认类型。', '依參數搜尋順序選擇元件；不選則使用預設型別。')}, function(_, index)
            if not index or not alive() or generation ~= revision then return end
            if index <= #document.definitions then
              local id, found = document.definitions[index].id, nil
              for i, value in ipairs(sources) do if value == id then found = i end end
              if found then table.remove(sources, found) else sources[#sources + 1] = id end
              return choose_sources()
            end
            if actions.dtype_sequence_guard and index == #document.definitions + 1 then
              tracking = not tracking; return choose_sources()
            end
            if index == apply_index then
              patch_node(node, {definition = vim.NIL, dtype_definitions = #sources > 0 and sources or vim.NIL,
                track_sequence = tracking and true or vim.NIL})
            end
          end)
        end
        choose_sources()
      elseif action == 'output_name' then
        vim.ui.input({prompt = tr('External output name: ', '对外输出名称：', '對外輸出名稱：'), default = node.output_name or node.id}, function(raw)
          if raw == nil or not alive() or generation ~= revision or find_node(node.id) ~= node then return end
          local name = vim.trim(raw)
          if not valid_id(name) then message = tr('Enter a valid output name.', '请输入有效输出名称。', '請輸入有效輸出名稱。'); return render() end
          for _, other in ipairs(document.nodes) do
            if other ~= node and other.kind == 'output' and (other.output_name or other.id) == name then
              message = tr('Output name already exists.', '输出名称已存在。', '輸出名稱已存在。'); return render()
            end
          end
          patch_node(node, {output_name = name == node.id and vim.NIL or name}, function()
            message = tr('Output name changed; check data, loss and supervision bindings.', '输出名称已变更，请核对数据、损失和监督绑定。', '輸出名稱已變更，請核對資料、損失和監督繫結。'); render()
          end)
        end)
      elseif action == 'flat_width' then
        vim.ui.input({prompt = tr('Expected output width: ', '预期输出宽度：', '預期輸出寬度：'), default = tostring(node.width or '')}, function(raw)
          if not raw or not alive() or generation ~= revision then return end
          local value = raw:match('^%d+$') and tonumber(raw) or nil
          if not value or value < 1 or value > 65536 then
            message = tr('Width must be between 1 and 65536.', '宽度须为 1 至 65536。', '寬度須為 1 至 65536。'); return render()
          end
          patch_node(node, {width = value})
        end)
      elseif action == 'dynamic_axes' then
        local axes = vim.deepcopy(node.axes or {})
        local function choose_axes()
          if not alive() or generation ~= revision or find_node(node.id) ~= node then return end
          local labels = {}
          for axis = 0, 7 do labels[#labels + 1] = (vim.tbl_contains(axes, axis) and '☑ ' or '☐ ') .. tostring(axis) end
          labels[#labels + 1] = tr('Apply', '应用', '套用')
          labels[#labels + 1] = tr('Cancel', '取消', '取消')
          vim.ui.select(labels, {prompt = tr('Sample axes (excluding batch)', '样本轴（不含批次轴）', '樣本軸（不含批次軸）')}, function(_, selected_axis)
            if not selected_axis or not alive() or generation ~= revision then return end
            if selected_axis <= 8 then
              local axis, index = selected_axis - 1, nil
              for i, value in ipairs(axes) do if value == axis then index = i end end
              if index then table.remove(axes, index) else axes[#axes + 1] = axis; table.sort(axes) end
              return choose_axes()
            end
            if selected_axis == 9 then
              if #axes == 0 then message = tr('Choose at least one axis.', '请至少选择一个轴。', '請至少選擇一個軸。'); return render() end
              patch_node(node, {axes = axes})
            end
          end)
        end
        choose_axes()
      elseif action == 'axis' then
        vim.ui.input({prompt = tr('Concatenation axis (integer): ', '拼接轴（整数）：', '串接軸（整數）：'), default = tostring(node.axis or '')}, function(raw)
          if not raw or not alive() or generation ~= revision then return end
          local value = raw:match('^%-?%d+$') and tonumber(raw) or nil
          if not value or value < -2147483648 or value > 2147483647 then message = tr('Invalid axis.', '拼接轴无效。', '串接軸無效。'); return render() end
          patch_node(node, {axis = value})
        end)
      end
    end
    if requested_action then dispatch() else vim.ui.select(choices, {prompt = node.id .. ' · ' .. node.kind}, dispatch) end
  end
  local function select_node(id)
    local node = find_node(id); if not node then return end
    selected, selected_edge = id, nil
    if pending then connect_finish(node) else render() end
  end
  local function move_node(id, destination)
    local function membership(node_id)
      for _, group in ipairs(document.groups or {}) do if vim.tbl_contains(group.nodes, node_id) then return group.id end end
    end
    if membership(id) ~= membership(destination) then
      message = tr('Drop on a group heading to change membership.', '拖到分组标题可更改归属。', '拖至分組標題可變更歸屬。'); return render()
    end
    local from, target
    for index, node_id in ipairs(layout_order) do
      if node_id == id then from = index end
      if node_id == destination then target = index end
    end
    if not from or not target or from == target then return end
    mutate(function()
      local node_id = table.remove(layout_order, from)
      table.insert(layout_order, target, node_id)
    end)
    message = tr('Display order changed; tensor connections are unchanged.', '显示顺序已调整；张量连接未变化。', '顯示順序已調整；張量連線未變更。')
    render()
    if selected_row and alive() then
      api.nvim_win_call(win, function()
        local outside = selected_row < vim.fn.line('w0') or selected_row > vim.fn.line('w$')
        api.nvim_win_set_cursor(win, {selected_row, 0})
        if outside then vim.cmd('normal! zz') end
      end)
    end
  end
  local function auto_layout()
    if pending or dragging then
      message = tr('Finish or cancel the current connection first.', '请先完成或取消当前接线。', '請先完成或取消目前的接線。'); return render()
    end
    local degree, outgoing, rank = {}, {}, {}
    for index, id in ipairs(layout_order) do degree[id], outgoing[id], rank[id] = 0, {}, index end
    if #layout_order ~= #document.nodes then
      message = tr('Resolve duplicate node IDs before arranging.', '请先处理重复节点 ID。', '請先處理重複節點 ID。'); return render()
    end
    for _, edge in ipairs(document.edges) do
      local from, to = edge.from.node, edge.to.node
      if degree[from] == nil or degree[to] == nil then
        message = tr('Resolve missing edge endpoints before arranging.', '请先处理缺失的连接端点。', '請先處理缺少的連線端點。'); return render()
      end
      outgoing[from][#outgoing[from] + 1] = to
      degree[to] = degree[to] + 1
    end
    local ready, order = {}, {}
    for _, id in ipairs(layout_order) do if degree[id] == 0 then ready[#ready + 1] = id end end
    while #ready > 0 do
      table.sort(ready, function(a, b) return rank[a] < rank[b] end)
      local id = table.remove(ready, 1); order[#order + 1] = id
      for _, next_id in ipairs(outgoing[id]) do
        degree[next_id] = degree[next_id] - 1
        if degree[next_id] == 0 then ready[#ready + 1] = next_id end
      end
    end
    if #order ~= #layout_order then
      message = tr('Cannot arrange a cyclic graph; fix the connections first.', '无法排列循环图，请先修正连接。', '無法排列循環圖，請先修正連線。'); return render()
    end
    mutate(function() layout_order = order end)
    message = tr('Display follows dependencies; graph semantics are unchanged.', '显示顺序已按依赖排列；图语义未变化。', '顯示順序已按相依關係排列；圖語意未變更。')
    render()
    if selected_row and alive() then
      api.nvim_win_set_cursor(win, {selected_row, 0})
      api.nvim_win_call(win, function() vim.cmd('normal! zz') end)
    end
  end
  function render()
    if not alive() then return end
    local view = api.nvim_win_call(win, vim.fn.winsaveview)
    local lines = {tr(' GRAPH · design nodes and connections', ' 图 · 设计节点与连接', ' 圖 · 設計節點與連線')}
    toolbar_rows = {}
    local toolbar = {
      {'H', tr('Structure', '基础结构', '基礎結構')}, {'a', tr('Add', '添加', '新增')}, {'Enter', tr('Details', '详情', '詳情')},
      {'c', tr('Connect', '接线', '接線')}, {'x', tr('Remove edge', '移除连接', '移除連線')},
      {'u', tr('Undo', '撤销', '復原')}, {'Ctrl-R', tr('Redo', '重做', '重做')}, {'/', tr('Find', '查找', '尋找')},
      {'o', tr('Arrange', '排列', '排列')}, {'P', tr('Plane', '平面', '平面')}, {'z', tr('Focus', '局部', '局部')}, {'G', tr('Groups', '分组', '分組')}, {'v', tr('Plan', '规划', '規劃')}, {'s', tr('Save', '保存', '儲存')},
      {'r', tr('Compare', '比较', '比較')}, {'g', tr('Generate', '生成', '產生')},
      {'e', tr('Python', 'Python 环境', 'Python 環境')},
      {'b', tr('Data', '数据', '資料')}, {'p', tr('Forward', '前向', '前向')},
      {'d', tr('Locate failure', '定位失败', '定位失敗')},
      {'i', tr('Capture', '捕获输入', '擷取輸入')}, {'I', tr('Replay', '重放输入', '重播輸入')},
      {'T', tr('Training settings', '训练设置', '訓練設定')}, {'t', tr('Trial', '试跑', '試跑')},
      {'Y', tr('Train', '训练', '訓練')}, {'R', tr('Runs', '运行记录', '執行紀錄')},
      {'Q', tr('Pending request', '待确认请求', '待確認要求')}, {'q', tr('Back', '返回', '返回')},
    }
    local line, limit = ' ', math.max(1, api.nvim_win_get_width(win) - 1)
    for _, item in ipairs(toolbar) do
      local part = '[' .. item[1] .. '] ' .. item[2] .. '  '
      if line ~= ' ' and vim.fn.strdisplaywidth(line .. part) > limit then
        lines[#lines + 1] = line; toolbar_rows[#lines] = true; line = ' '
      end
      line = line .. part
    end
    if line ~= ' ' then lines[#lines + 1] = line; toolbar_rows[#lines] = true end
    lines[#lines + 1] = ''
    rows, edge_rows, port_hits, selected_row, selected_edge_row, group_rows = {}, {}, {}, nil, nil, {}
    if message then lines[#lines + 1] = '! ' .. message end
    if pending then lines[#lines + 1] = tr('Connecting from ', '接线来源：', '接線來源：') .. pending.node .. '.' .. pending.port end
    if document.supervision then
      lines[#lines + 1] = tr('Supervision reference: ', '监督引用：', '監督參照：')
        .. vim.fn.strcharpart(tostring(document.supervision.file):gsub('[%c]', ' '), 0, 160)
        .. tr(' · import and rebind [V]', ' · 导入并重绑定 [V]', ' · 匯入並重新繫結 [V]')
    end
    if document.training then
      local optimizer = document.training.optimizer or {}
      local mode = document.training.mode or 'managed-step'
      local name = mode == 'external-loop' and tr('loop owns optimizer', '循环负责优化器', '迴圈負責最佳化器')
        or optimizer.factory or tr('AdamW defaults/settings', 'AdamW 默认值或设置', 'AdamW 預設值或設定')
      lines[#lines + 1] = tr('Trial: ', '试跑：', '試跑：') .. (document.training.trial_steps and (tostring(document.training.trial_steps) .. tr(' steps', ' 步', ' 步')) or tr('steps not set', '步数未设置', '步數未設定')) .. ' · '
        .. mode .. ' · ' .. vim.fn.strcharpart(tostring(name):gsub('[%c]', ' '), 0, 120)
        .. ' · ' .. vim.fn.strcharpart(tostring(document.training.step or tr('generated step', '生成的训练步骤', '產生的訓練步驟')):gsub('[%c]', ' '), 0, 120)
        .. tr(' · editable [T]', ' · 可编辑 [T]', ' · 可編輯 [T]')
    end
    lines[#lines + 1] = tr('Drag nodes or ports; click an incoming edge to inspect.', '拖动节点或端口；点击入边查看操作。', '拖曳節點或連接埠；點選入邊查看操作。')
    local incoming, known_nodes, nodes_by_id, use_count = {}, {}, {}, {}
    for _, node in ipairs(document.nodes) do
      known_nodes[node.id] = true
      nodes_by_id[node.id] = node
      if node.definition then use_count[node.definition] = (use_count[node.definition] or 0) + 1 end
    end
    for _, edge in ipairs(document.edges) do
      incoming[edge.to.node] = incoming[edge.to.node] or {}
      incoming[edge.to.node][#incoming[edge.to.node] + 1] = edge
    end
    local visible
    if focus_root and known_nodes[focus_root] then
      visible = {[focus_root] = true}
      for _, edge in ipairs(document.edges) do
        if edge.from.node == focus_root then visible[edge.to.node] = true end
        if edge.to.node == focus_root then visible[edge.from.node] = true end
      end
      if selected and not visible[selected] then visible, focus_root = nil, nil end
    else focus_root = nil end
    if visible then
      local hidden = 0
      for _, id in ipairs(layout_order) do if not visible[id] then hidden = hidden + 1 end end
      lines[#lines + 1] = tr('Focused on ', '局部查看：', '局部檢視：') .. focus_root
        .. tr(' · hidden nodes: ', ' · 隐藏节点：', ' · 隱藏節點：') .. hidden
        .. tr(' · z to show all', ' · z 显示全部', ' · z 顯示全部')
    end
    local function edge_line(edge)
      lines[#lines + 1] = (edge == selected_edge and '  ▶ ' or '  └─ ') .. edge.from.node .. '.' .. edge.from.port .. ' → '
        .. (known_nodes[edge.to.node] and '' or (edge.to.node .. '.')) .. edge.to.port
        .. (visible and not visible[edge.from.node] and tr(' · outside view', ' · 在视图外', ' · 在視圖外') or '')
      edge_rows[#lines] = edge
      if edge == selected_edge then selected_edge_row = #lines end
    end
    local group_by_node, groups = {}, {}
    for _, group in ipairs(document.groups or {}) do
      groups[group.id] = group
      for _, id in ipairs(group.nodes) do if not group_by_node[id] then group_by_node[id] = group.id end end
    end
    if selected and group_by_node[selected] then
      local current, changed = group_by_node[selected], false
      while current and groups[current] do
        if collapsed_groups[current] then collapsed_groups[current], changed = nil, true end
        current = groups[current].parent
      end
      if changed then layout_changed() end
    end
    local display, entries, group_rank, group_count = {}, {[false] = {}}, {}, {}
    for id in pairs(groups) do entries[id] = {} end
    for index, id in ipairs(layout_order) do
      local group_id = group_by_node[id]
      if not visible or visible[id] then
        local parent = group_id or false
        entries[parent][#entries[parent] + 1] = {node = id, rank = index}
        while group_id and groups[group_id] do
          group_rank[group_id] = math.min(group_rank[group_id] or index, index)
          group_count[group_id] = (group_count[group_id] or 0) + 1
          group_id = groups[group_id].parent
        end
      end
    end
    for id, group in pairs(groups) do
      if group_rank[id] then
        local parent = group.parent or false
        entries[parent][#entries[parent] + 1] = {group = id, rank = group_rank[id]}
      end
    end
    local function append_entries(parent, depth)
      table.sort(entries[parent], function(a, b) return a.rank < b.rank end)
      for _, item in ipairs(entries[parent]) do
        item.depth = depth
        if item.node then display[#display + 1] = item
        else
          item.count = group_count[item.group]; display[#display + 1] = item
          if not collapsed_groups[item.group] then append_entries(item.group, depth + 1) end
        end
      end
    end
    append_entries(false, 0)
    for _, entry in ipairs(display) do
      if entry.group then
        lines[#lines + 1] = string.rep(' ', math.min(entry.depth, 12) * 2) .. (collapsed_groups[entry.group] and '  ▸ ' or '  ▾ ') .. entry.group
          .. ' · ' .. entry.count .. tr(' nodes', ' 个节点', ' 個節點')
          .. (collapsed_groups[entry.group] and tr(' · edges hidden', ' · 连接已折叠', ' · 連線已摺疊') or '')
        group_rows[#lines] = entry.group
      else
      local node_id = entry.node
      local node = nodes_by_id[node_id]
      if not visible or visible[node_id] then
      for _, edge in ipairs(incoming[node.id] or {}) do edge_line(edge) end
      local line = string.rep(' ', math.min(entry.depth, 12) * 2) .. (node.id == selected and '▶ ' or '  ') .. node.id .. '  [' .. node.kind .. ']  '
      local hits = {}
      local function add_ports(ports, direction, arrow)
        if #ports == 0 then return end
        line = line .. arrow
        for _, port in ipairs(ports) do
          local start = #line
          line = line .. '[' .. port .. ']'
          hits[#hits + 1] = {start = start, finish = #line, node = node.id, port = port, direction = direction}
        end
        line = line .. ' '
      end
      add_ports(node.inputs or {}, 'input', '← ')
      add_ports(node.outputs or {}, 'output', '→ ')
      line = line .. (node.definition and (' · ' .. node.definition .. ((use_count[node.definition] or 0) > 1
        and (' ×' .. tostring(use_count[node.definition])) or '')) or '')
      lines[#lines + 1] = line
      rows[#lines], port_hits[#lines] = node.id, hits
      if node.id == selected then
        selected_row = #lines
        local function clean(value)
          local text = tostring(value or '?'):gsub('[%c]', ' ')
          return vim.fn.strcharpart(text, 0, 180) .. (vim.fn.strchars(text) > 180 and '…' or '')
        end
        local function encode(value)
          local ok, raw = pcall(vim.json.encode, value)
          return ok and clean(raw) or '?'
        end
        if node.kind == 'call' then
          local definition = find_definition(node.definition)
          if definition then
            lines[#lines + 1] = '    ' .. tr('Factory: ', '工厂：', '工廠：') .. clean(definition.factory)
            local names = vim.tbl_keys(definition.params or {}); table.sort(names)
            local parts = {}
            for index, name in ipairs(names) do
              if index > 3 then parts[#parts + 1] = '…'; break end
              parts[#parts + 1] = clean(name) .. '=' .. encode(definition.params[name])
            end
            lines[#lines + 1] = '    ' .. tr('Parameters: ', '参数：', '參數：') .. (#parts > 0 and table.concat(parts, ', ') or '∅')
              .. ' · ' .. tr('Call: ', '调用：', '呼叫：') .. (node.call_style or 'keyword')
          end
        elseif node.kind == 'concat' then
          lines[#lines + 1] = '    ' .. tr('Axis: ', '拼接轴：', '串接軸：') .. (node.axis ~= nil and tostring(node.axis) or '?')
        elseif node.kind == 'output' then
          local summaries = {}
          for _, objective in ipairs(document.objectives or {}) do
            if objective.output == node.id then
              summaries[#summaries + 1] = clean(objective.loss) .. ' → ' .. clean(objective.target) .. ' × ' .. tostring(objective.weight)
            end
          end
          lines[#lines + 1] = '    ' .. tr('Losses: ', '损失：', '損失：')
            .. (#summaries > 0 and clean(table.concat(summaries, ', ')) or tr('not declared', '未声明', '未宣告'))
        end
        local shapes = {}
        for _, group in ipairs({'input_shapes', 'output_shapes'}) do
          local names = vim.tbl_keys(node[group] or {}); table.sort(names)
          for _, name in ipairs(names) do
            if #shapes >= 3 then break end
            shapes[#shapes + 1] = (group == 'input_shapes' and '← ' or '→ ') .. clean(name) .. ' ' .. encode(node[group][name])
          end
        end
        if #shapes > 0 then lines[#lines + 1] = '    ' .. tr('Declared shapes: ', '声明形状：', '宣告形狀：') .. table.concat(shapes, ' · ') end
        local dtypes = {}
        for _, group in ipairs({'input_dtypes', 'output_dtypes'}) do
          local names = vim.tbl_keys(node[group] or {}); table.sort(names)
          for _, name in ipairs(names) do
            if #dtypes >= 3 then break end
            dtypes[#dtypes + 1] = (group == 'input_dtypes' and '← ' or '→ ') .. clean(name) .. ' ' .. clean(node[group][name])
          end
        end
        if #dtypes > 0 then lines[#lines + 1] = '    ' .. tr('Declared dtypes: ', '声明 dtype：', '宣告 dtype：') .. table.concat(dtypes, ' · ') end
        if node.kind == 'input' or node.kind == 'output' then
          local marks = {}
          for port, label in pairs(node.semantics or {}) do marks[#marks + 1] = clean(port) .. ': ' .. clean(label) end
          table.sort(marks)
          if #marks > 0 then lines[#lines + 1] = '    ' .. tr('Semantic labels: ', '语义标记：', '語意標記：') .. clean(table.concat(marks, ' · '))
            .. tr(' · declared only', ' · 仅声明', ' · 僅宣告') end
          local draft_count = 0; for _, rules in pairs(node.draft_rules or {}) do draft_count = draft_count + #rules end
          if draft_count > 0 then lines[#lines + 1] = '    ' .. tr('Draft rules (not bound or enforced): ', '草稿规则（未绑定、未执行）：', '草稿規則（未繫結、未執行）：') .. draft_count end
          lines[#lines + 1] = '    ' .. tr('Data: ', '数据：', '資料：') .. (document.data and clean(document.data.factory)
            or tr('not declared', '未声明', '未宣告')) .. tr(' · not runtime verified', ' · 尚未运行验证', ' · 尚未執行驗證')
        end
      end
      end
      end
    end
    if #document.nodes == 0 then lines[#lines + 1] = tr('  Choose a basic structure [H] or add nodes [a] to begin.', '  选择基础结构 [H] 或添加节点 [a] 开始设计。', '  選擇基礎結構 [H] 或新增節點 [a] 開始設計。') end
    local unresolved = {}
    for _, edge in ipairs(document.edges) do if not known_nodes[edge.to.node] then unresolved[#unresolved + 1] = edge end end
    if #unresolved > 0 then
      lines[#lines + 1] = ''; lines[#lines + 1] = tr('Connections with missing destinations', '目标节点缺失的连接', '目標節點缺失的連線')
      for _, edge in ipairs(unresolved) do edge_line(edge) end
    end
    lines[#lines + 1] = ''; lines[#lines + 1] = tr('Recovery is separate from saving. Shapes and factories are unverified until run.',
      '恢复文件不等于保存项目。形状和工厂在运行前未经验证。', '復原檔案不等於儲存專案。形狀和工廠在執行前未經驗證。')
    vim.bo[buf].modifiable = true; api.nvim_buf_set_lines(buf, 0, -1, false, lines); vim.bo[buf].modifiable = false
    api.nvim_buf_clear_namespace(buf, ns, 0, -1)
    if selected_row then api.nvim_buf_set_extmark(buf, ns, selected_row - 1, 0, {line_hl_group = 'PmenuSel'}) end
    if selected_edge_row then api.nvim_buf_set_extmark(buf, ns, selected_edge_row - 1, 0, {line_hl_group = 'PmenuSel'}) end
    if dragging and dragging.target then
      if type(dragging.target) == 'table' and dragging.target.group then
        for row, id in pairs(group_rows) do if id == dragging.target.group then
          api.nvim_buf_set_extmark(buf, ns, row - 1, 0, {line_hl_group = 'Search'}); break
        end end
      else
        local target_id = type(dragging.target) == 'table' and dragging.target.node or dragging.target
        for row, id in pairs(rows) do if id == target_id then
          local conflict = dragging.mode == 'wire' and shape_conflict(dragging.source, find_node(target_id), dragging.target.port)
          api.nvim_buf_set_extmark(buf, ns, row - 1, 0, {line_hl_group = conflict and 'ErrorMsg' or 'Search'}); break
        end end
      end
    end
    view.lnum = math.min(view.lnum, #lines)
    view.topline = math.min(view.topline, #lines)
    api.nvim_win_call(win, function() vim.fn.winrestview(view) end)
  end
  local function find_nodes()
    if pending or dragging then
      message = tr('Finish or cancel the current connection first.', '请先完成或取消当前接线。', '請先完成或取消目前的接線。')
      return render()
    end
    local revision = generation
    vim.ui.input({prompt = tr('Find node, kind, definition, factory or port: ', '查找节点、类型、组件、工厂或端口：', '尋找節點、類型、元件、工廠或連接埠：')}, function(query)
      if not query or not alive() or revision ~= generation then return end
      query = vim.trim(query):lower()
      if query == '' then return end
      local matches = {}
      for _, node in ipairs(document.nodes) do
        local definition = node.definition and find_definition(node.definition)
        local description = table.concat({node.id, node.kind, node.definition or '', definition and definition.factory or '',
          table.concat(node.inputs or {}, ' '), table.concat(node.outputs or {}, ' ')}, ' ')
        if description:lower():find(query, 1, true) then
          matches[#matches + 1] = {id = node.id, label = node.id .. '  [' .. node.kind .. ']'
            .. (node.definition and ('  · ' .. node.definition) or '')}
        end
      end
      if #matches == 0 then
        message = tr('No matching nodes.', '没有匹配的节点。', '沒有符合的節點。'); return render()
      end
      vim.ui.select(matches, {prompt = tr('Select a matching node', '选择匹配的节点', '選擇符合的節點'),
        format_item = function(item) return item.label end}, function(item)
        if not item or not alive() or revision ~= generation or not find_node(item.id) then return end
        selected, selected_edge, message = item.id, nil, nil
        render()
        if selected_row then
          api.nvim_win_set_cursor(win, {selected_row, 0})
          api.nvim_win_call(win, function() vim.cmd('normal! zz') end)
        end
      end)
    end)
  end
  local function invoke(action)
    if not actions[action] then return end
    local snapshot = vim.deepcopy(document)
    actions[action](snapshot, function(info, focus, merged)
      if not alive() then return false end
      if merged then
        if not vim.deep_equal(document, snapshot) then return false end
        mutate(function() replace(merged) end)
        if selected and not find_node(selected) then selected = nil end
      end
      local valid_focus = focus and find_node(focus) and vim.deep_equal(document, snapshot)
      if valid_focus then
        selected, selected_edge = focus, nil
      end
      message = info; render()
      if plane_status then plane_status(info, valid_focus and focus or nil) end
      if valid_focus and selected_row then
        api.nvim_win_set_cursor(win, {selected_row, 0})
        api.nvim_win_call(win, function() vim.cmd('normal! zz') end)
      end
      if action == 'locate_failure' and valid_focus then
        local revision, id = generation, focus
        local node = find_node(id)
        local entries = {{tr('Inspect input connections', '检查输入连接', '檢查輸入連線'), 'inputs'},
          {tr('Open node details', '打开节点详情', '開啟節點詳情'), 'details'},
          {tr('Open generated code (saved graph)', '打开生成代码（已保存图）', '開啟產生的程式碼（已儲存圖）'), 'source'},
          {tr('Replay captured input', '重放已捕获输入', '重播已擷取輸入'), 'replay'},
          {tr('Keep selection', '保留选中状态', '保留選取狀態'), 'cancel'}}
        if node.kind == 'call' then table.insert(entries, 2, {tr('Edit component parameter', '编辑组件参数', '編輯元件參數'), 'parameter'}) end
        local labels = {}; for _, entry in ipairs(entries) do labels[#labels + 1] = entry[1] end
        vim.ui.select(labels, {prompt = tr('Possible failure location: ', '可能出错的位置：', '可能出錯的位置：') .. id
          .. tr(' · choose a next step', ' · 选择下一步', ' · 選擇下一步')}, function(_, choice)
          if not choice or not alive() or generation ~= revision or not vim.deep_equal(document, snapshot) or selected ~= id then return end
          local next_action = entries[choice][2]
          if next_action == 'inputs' then
            local edges, names = {}, {}
            for _, edge in ipairs(document.edges) do
              if edge.to.node == id then
                edges[#edges + 1] = edge
                names[#names + 1] = edge.from.node .. '.' .. edge.from.port .. ' → ' .. edge.to.node .. '.' .. edge.to.port
              end
            end
            if #edges == 0 then message = tr('No input connections are present.', '没有输入连接。', '沒有輸入連線。'); return render() end
            vim.ui.select(names, {prompt = tr('Inspect an input connection', '选择要检查的输入连接', '選擇要檢查的輸入連線')}, function(_, index)
              if index and alive() and generation == revision and vim.deep_equal(document, snapshot) and selected == id then edge_details(edges[index]) end
            end)
          elseif next_action == 'details' then details()
          elseif next_action == 'parameter' then edit_parameter(node)
          elseif next_action == 'source' then details('generated_source')
          elseif next_action == 'replay' then invoke('replay_input') end
        end)
      end
      return true
    end)
  end
  local function configure_training(import_session)
    local edit = import_session and actions.import_training or actions.configure_training
    if not edit then return end
    local snapshot, revision = vim.deepcopy(document), generation
    edit(snapshot, function(training)
      if not training or not alive() or generation ~= revision or not vim.deep_equal(document, snapshot) then return end
      patch_graph({training = training}, function()
        message = tr('Training settings saved in draft; save the graph before a trial.', '训练设置已写入草稿；试跑前请保存图。', '訓練設定已寫入草稿；試跑前請儲存圖。'); render()
      end)
    end)
  end
  local function link_supervision()
    if not actions.link_supervision then
      message = tr('This CLI cannot save supervision references.', '当前 CLI 不支持保存监督引用。', '目前 CLI 不支援儲存監督參照。'); return render()
    end
    local snapshot, revision = vim.deepcopy(document), generation
    actions.link_supervision(snapshot, function(reference)
      if not reference or not alive() or generation ~= revision or not vim.deep_equal(document, snapshot) then return end
      patch_graph({supervision = reference}, function()
        message = tr('Reference added to draft; save the graph before packaging or importing.',
          '引用已加入草稿；打包或导入前请保存图。', '參照已加入草稿；打包或匯入前請儲存圖。'); render()
      end)
    end)
  end
  local function remove_supervision_reference()
    if not document.supervision then return end
    local revision = generation
    vim.ui.select({tr('Keep', '保留', '保留'), tr('Remove reference', '移除引用', '移除參照')}, {
      prompt = tr('Only the graph reference is removed; the file and bound rules remain.',
        '只移除图中的引用；文件和已绑定规则保留。', '只移除圖中的參照；檔案和已繫結規則保留。'),
    }, function(_, choice)
      if choice == 2 and alive() and generation == revision then patch_graph({supervision = vim.NIL}) end
    end)
  end
  local function configure_component_storage()
    if not actions.named_components then
      message = tr('The selected CLI does not support named component storage.', '所选 CLI 不支持具名组件存储。', '所選 CLI 不支援具名元件儲存。'); return render()
    end
    local revision = generation
    vim.ui.select({tr('Stable names based on component IDs', '基于组件 ID 的稳定名称', '基於元件 ID 的穩定名稱'),
      tr('Legacy positional names', '旧版位置名称', '舊版位置名稱'), tr('Cancel', '取消', '取消')}, {
        prompt = tr('Current: ', '当前：', '目前：') .. (document.component_storage == 'named'
          and tr('stable names', '稳定名称', '穩定名稱') or tr('legacy positions', '旧版位置', '舊版位置'))
          .. tr('\nChanging this changes checkpoint keys. Regenerate code and migrate weights explicitly.',
            '\n切换会改变检查点参数名。需要重新生成代码并显式迁移权重。',
            '\n切換會改變檢查點參數名稱。需要重新產生程式碼並明確遷移權重。'),
      }, function(_, choice)
        if not choice or choice == 3 or not alive() or generation ~= revision then return end
        local selected = choice == 1 and 'named' or nil
        if document.component_storage == selected then return end
        patch_graph({component_storage = selected or vim.NIL}, function()
          message = tr('Naming changed in draft; regenerate and migrate weights separately.',
            '命名已在草稿中修改；请另行生成代码和迁移权重。', '命名已在草稿中修改；請另行產生程式碼和遷移權重。'); render()
        end)
      end)
  end
  local function reorder_chain(source_id, target_id, after, plane_positions, done)
    local revision, incoming, outgoing = generation, {}, {}
    for _, edge in ipairs(document.edges) do
      incoming[edge.to.node] = incoming[edge.to.node] or {}; incoming[edge.to.node][#incoming[edge.to.node] + 1] = edge
      outgoing[edge.from.node] = outgoing[edge.from.node] or {}; outgoing[edge.from.node][#outgoing[edge.from.node] + 1] = edge
    end
    local function simple(id)
      local node, ins, outs = find_node(id), incoming[id] or {}, outgoing[id] or {}
      return node and node.kind == 'call' and #node.inputs == 1 and #node.outputs == 1 and #ins == 1 and #outs == 1
        and ins[1].to.port == node.inputs[1] and outs[1].from.port == node.outputs[1]
    end
    if source_id == target_id or not simple(source_id) or not simple(target_id) then
      return done(nil, nil, tr('Reordering needs two single-input/output calls in the same unbranched chain.', '重排需要同一无分支链中的两个单输入输出组件。', '重排需要同一無分支鏈中的兩個單輸入輸出元件。'))
    end
    local first, visited = source_id, {}
    while true do
      if visited[first] then return done(nil, nil, tr('The chain contains a cycle.', '链中存在循环。', '鏈中存在循環。')) end
      visited[first] = true
      local previous = incoming[first][1].from.node
      if not simple(previous) then break end
      first = previous
    end
    local chain, current, seen = {}, first, {}
    while simple(current) do
      if seen[current] then return done(nil, nil, tr('The chain contains a cycle.', '链中存在循环。', '鏈中存在循環。')) end
      seen[current], chain[#chain + 1] = true, current
      current = outgoing[current][1].to.node
    end
    if not seen[target_id] then return done(nil, nil, tr('Nodes are not in the same unbranched chain.', '节点不在同一无分支链中。', '節點不在同一無分支鏈中。')) end
    local membership = {}
    for _, group in ipairs(document.groups or {}) do
      for _, id in ipairs(group.nodes or {}) do membership[id] = group.id end
    end
    local crosses_groups = false
    for _, id in ipairs(chain) do if membership[id] ~= membership[source_id] then crosses_groups = true; break end end
    local reordered = vim.deepcopy(chain)
    local from = vim.fn.index(reordered, source_id) + 1
    table.remove(reordered, from)
    local target = vim.fn.index(reordered, target_id) + 1
    table.insert(reordered, target + (after and 1 or 0), source_id)
    if vim.deep_equal(chain, reordered) then return done(nil, nil, tr('The node is already at that position.', '该节点已位于指定位置。', '該節點已位於指定位置。')) end
    local removed, entry, exit = {}, incoming[first][1], outgoing[chain[#chain]][1]
    removed[entry] = true
    for _, id in ipairs(chain) do removed[outgoing[id][1]] = true end
    local new_edges, first_conflict = {}, nil
    local function append(from_node, from_port, to_node, to_port)
      local src, dst = find_node(from_node), find_node(to_node)
      if not src or not dst or not vim.tbl_contains(src.outputs, from_port) or not vim.tbl_contains(dst.inputs, to_port)
          or shape_conflict({node = from_node, port = from_port}, dst, to_port) then
        first_conflict = first_conflict or (from_node .. '.' .. from_port .. ' → ' .. to_node .. '.' .. to_port)
        return false
      end
      new_edges[#new_edges + 1] = {from = {node = from_node, port = from_port}, to = {node = to_node, port = to_port}}
      return true
    end
    local new_first, new_last = find_node(reordered[1]), find_node(reordered[#reordered])
    local valid = append(entry.from.node, entry.from.port, new_first.id, new_first.inputs[1])
    for index = 1, #reordered - 1 do
      local left, right = find_node(reordered[index]), find_node(reordered[index + 1])
      valid = append(left.id, left.outputs[1], right.id, right.inputs[1]) and valid
    end
    valid = append(new_last.id, new_last.outputs[1], exit.to.node, exit.to.port) and valid
    if not valid then return done(nil, nil, tr('A new connection conflicts with a declared shape or port: ', '新连接与声明形状或端口冲突：', '新連線與宣告形狀或連接埠衝突：') .. (first_conflict or '?')) end
    local dtype_hints, dtype_count = {}, 0
    for _, edge in ipairs(new_edges) do
      local from, to = find_node(edge.from.node), find_node(edge.to.node)
      local a = from and from.output_dtypes and from.output_dtypes[edge.from.port]
      local b = to and to.input_dtypes and to.input_dtypes[edge.to.port]
      if a and b and a ~= b then
        dtype_count = dtype_count + 1
        if #dtype_hints < 4 then
          dtype_hints[#dtype_hints + 1] = edge.from.node .. '.' .. edge.from.port .. ' (' .. a .. ') → '
            .. edge.to.node .. '.' .. edge.to.port .. ' (' .. b .. ')'
        end
      end
    end
    if not actions.plan_edits and not actions.plan_reorder then return done(nil, nil, tr('Graph reorder planning is unavailable.', '图重排规划不可用。', '圖重排規劃不可用。')) end
    local shared, shared_seen = {}, {}
    for _, id in ipairs(chain) do
      local node = find_node(id)
      local definition = node and node.definition
      if definition and not shared_seen[definition] then
        shared_seen[definition] = true
        local count = 0
        for _, call in ipairs(document.nodes) do if call.definition == definition then count = count + 1 end end
        if count > 1 then shared[#shared + 1] = definition .. ' ×' .. count end
      end
    end
    local snapshot, candidate = vim.deepcopy(document), vim.deepcopy(document)
    for index = #document.edges, 1, -1 do if removed[document.edges[index]] then table.remove(candidate.edges, index) end end
    for _, edge in ipairs(new_edges) do candidate.edges[#candidate.edges + 1] = vim.deepcopy(edge) end
    local dtype_sequence_updated = false
    for _, cast in ipairs(document.nodes) do
      if actions.reorder_dtype_sync and cast.kind == 'cast' and cast.track_sequence then
        local cursor, relevant = cast.id, false
        for _ = 1, #document.nodes do
          local links = outgoing[cursor] or {}
          if #links ~= 1 then break end
          local next_id = links[1].to.node
          if next_id == chain[1] then relevant = true; break end
          local next_node = find_node(next_id)
          if not next_node or not vim.tbl_contains({'call', 'assert_flat'}, next_node.kind) then break end
          cursor = next_id
        end
        if relevant then
          dtype_sequence_updated = true
          local before, after_defs = {}, {}
          for _, id in ipairs(chain) do before[#before + 1] = find_node(id).definition end
          for _, id in ipairs(reordered) do after_defs[#after_defs + 1] = find_node(id).definition end
          local start
          for index = 1, #(cast.dtype_definitions or {}) - #before + 1 do
            local equal = true
            for offset, id in ipairs(before) do
              if cast.dtype_definitions[index + offset - 1] ~= id then equal = false; break end
            end
            if equal then start = index; break end
          end
          if not start then return done(nil, nil, tr('Dtype source order is stale; review it first.',
            '类型来源顺序已过期，请先检查。', '型別來源順序已過期，請先檢查。')) end
          for _, item in ipairs(candidate.nodes) do
            if item.id == cast.id then
              for offset, id in ipairs(after_defs) do item.dtype_definitions[start + offset - 1] = id end
              break
            end
          end
        end
      end
    end
    local function review(proposal, error)
      if not alive() or generation ~= revision or not vim.deep_equal(document, snapshot) then return done(nil, nil, tr('Graph changed; plan again.', '图已变化，请重新规划。', '圖已變更，請重新規劃。')) end
      if error or not proposal then return done(nil, nil, error) end
      if not vim.deep_equal(candidate, proposal) then return done(nil, nil, tr('Backend proposal differs; no edit was applied.', '后端方案不一致，未应用修改。', '後端方案不一致，未套用修改。')) end
      vim.ui.select({tr('Apply order', '应用顺序', '套用順序'), tr('Cancel', '取消', '取消')}, {
        prompt = tr('Execution order: ', '执行顺序：', '執行順序：') .. table.concat(chain, ' → ')
          .. tr('  becomes  ', '  改为  ', '  改為  ') .. table.concat(reordered, ' → ')
          .. tr('\nThe corresponding blocks will move to show the new order.', '\n对应的块也会移动，以显示新顺序。', '\n對應的區塊也會移動，以顯示新順序。')
          .. (crosses_groups and tr('\nDisplay group membership remains unchanged.', '\n显示分组的成员关系保持不变。', '\n顯示分組的成員關係保持不變。') or '')
          .. (#shared > 0 and (tr('\nShared definitions retain their instances: ', '\n共享定义保持同一实例：', '\n共用定義保持同一執行個體：') .. table.concat(shared, ', ')) or '')
          .. (dtype_sequence_updated and tr('\nThe tracked dtype source order will also change.',
            '\n跟踪的类型来源顺序也会同步调整。', '\n追蹤的型別來源順序也會同步調整。') or '')
          .. (dtype_count > 0 and ('\n' .. tr('Review dtype names (may be aliases; no conversion): ',
            '请核对 dtype 名称（可能是别名，不自动转换）：', '請核對 dtype 名稱（可能是別名，不自動轉換）：')
            .. table.concat(dtype_hints, '; ') .. (dtype_count > #dtype_hints and (' … +' .. (dtype_count - #dtype_hints)) or '')) or ''),
      }, function(_, choice)
        if choice ~= 1 or not alive() or generation ~= revision or not vim.deep_equal(document, snapshot) then return done(nil, nil, nil) end
        mutate(function()
          replace(candidate)
          if type(plane_positions) == 'table' then
            local slots = {}
            for index, id in ipairs(chain) do slots[index] = vim.deepcopy(plane_positions[id]) end
            local complete = true
            for _, point in ipairs(slots) do if type(point) ~= 'table' then complete = false; break end end
            if #slots ~= #chain then complete = false end
            if complete then
              layout_positions = vim.deepcopy(plane_positions)
              for index, id in ipairs(reordered) do layout_positions[id] = slots[index] end
            end
          end
          local members, position = {}, 0
          for _, id in ipairs(chain) do members[id] = true end
          for index, id in ipairs(layout_order) do
            if members[id] then position = position + 1; layout_order[index] = reordered[position] end
          end
        end)
        done(vim.deepcopy(document), vim.deepcopy(layout_positions))
      end)
    end
    if actions.plan_edits then
      actions.plan_edits(snapshot, {{action = after and 'move_after' or 'move_before', node = source_id, anchor = target_id}}, review)
    else
      actions.plan_reorder(snapshot, source_id, target_id, after, review)
    end
  end
  local function open_plane()
    if pending or dragging then message = tr('Finish or cancel the current gesture first.', '请先完成或取消当前操作。', '請先完成或取消目前操作。'); return render() end
    local expected = generation
    local restored_view = vim.deepcopy(plane_view or {topline = 1, leftcol = 0})
    if selected and find_node(selected) then restored_view.selected = selected end
    local controller
    controller = require('model_maven.graph_plane').open(vim.deepcopy(document), vim.deepcopy(layout_positions), {
      tr = tr,
      preview_boundary = actions.preview_boundary,
      view_state = restored_view,
      data_status = actions.data_status,
      source_status = actions.source_status,
      view_changed = function(view, force)
        plane_view = vim.deepcopy(view)
        if view.selected and find_node(view.selected) then selected, selected_edge = view.selected, nil end
        if actions.view_changed then actions.view_changed(view, force) end
      end,
      reorder_after = actions.reorder_after == true,
      delivery_review = actions.delivery_review ~= nil,
      delivery_resume = actions.delivery_resume ~= nil,
      is_saved = function(snapshot)
        return actions.is_saved and actions.is_saved(snapshot) or false
      end,
      layout = function(split, source_right)
        if not alive() then return end
        code_layout = split
        if source_right ~= nil then source_on_right = source_right end
        local preview_width, preview_col = require('model_maven.model_preview').layout(110, actions.preview_boundary)
        local next_width = split and math.max(1, math.floor((vim.o.columns - 5) / 2)) or preview_width
        local next_height = math.max(1, math.min(34, vim.o.lines - 5))
        pcall(api.nvim_win_set_config, win, {relative = 'editor', row = 1,
          col = split and (source_on_right and 1 or math.max(0, vim.o.columns - next_width - 2)) or preview_col,
          width = next_width, height = next_height})
        render()
      end,
      move = function(positions)
        if not alive() or generation ~= expected then return false end
        mutate(function() layout_positions = vim.deepcopy(positions) end)
        expected = generation
        return true
      end,
      history = function(redo_action)
        if not alive() or generation ~= expected then return nil, nil, 'stale' end
        local prior = generation
        restore(redo_action)
        if generation == prior then return nil, nil, 'empty' end
        expected = generation
        return vim.deepcopy(document), vim.deepcopy(layout_positions)
      end,
      reorder = function(source, target, after, positions, done)
        if not alive() or generation ~= expected then return done(nil, nil, tr('Graph changed; reopen this view.', '图已变化，请重新打开此视图。', '圖已變更，請重新開啟此檢視。')) end
        reorder_chain(source, target, after, positions, function(next_doc, next_positions, error)
          if next_doc then expected = generation end
          done(next_doc, next_positions, error)
        end)
      end,
      command = function(command, focused_node)
        if not alive() or generation ~= expected then return end
        if command == 'add_operator' then
          local outgoing = {}
          for _, edge in ipairs(document.edges) do if edge.from.node == focused_node then outgoing[#outgoing + 1] = edge end end
          if not focused_node and #document.edges == 1 then outgoing[1] = document.edges[1] end
          if #outgoing == 1 then
            add_node(outgoing[1])
          elseif #outgoing > 1 then
            local labels = {tr('Add an unconnected node', '添加未连接节点', '新增未連接節點')}
            for _, edge in ipairs(outgoing) do labels[#labels + 1] = edge.from.port .. ' → ' .. edge.to.node .. '.' .. edge.to.port end
            local revision = generation
            vim.ui.select(labels, {prompt = tr('Insert after the selected node on a connection', '选择所选节点之后的连接并插入', '選擇所選節點之後的連線並插入')}, function(_, choice)
              if not choice or not alive() or generation ~= revision then return end
              if choice == 1 then add_node(nil, 'call') else add_node(outgoing[choice - 1]) end
            end)
          else add_node(nil, 'call') end
        elseif command == 'add' then add_node()
        elseif command == 'groups' then manage_groups()
        elseif command == 'structure' then add_structure()
        elseif command == 'training' then configure_training()
        elseif command == 'training_import' then configure_training(true)
        elseif command == 'link_supervision' then link_supervision()
        elseif command == 'remove_supervision' then remove_supervision_reference()
        elseif command == 'component_storage' then configure_component_storage()
        elseif command == 'source' and actions.workspace_source then
          local snapshot = vim.deepcopy(document)
          actions.workspace_source(snapshot, function(source_right)
            if not alive() or generation ~= expected or not vim.deep_equal(document, snapshot) then return end
            show_source_side(source_right)
          end)
        elseif command == 'source_check' then
          vim.ui.select({tr('Check current project source', '核对当前项目源码', '核對目前專案原始碼'),
            tr('Generate or update source', '生成或更新源码', '產生或更新原始碼'), tr('Cancel', '取消', '取消')}, {
            prompt = tr('The graph source has not been checked against this project.', '尚未核对图与此项目的源码。', '尚未核對圖與此專案的原始碼。'),
          }, function(_, choice)
            if not alive() or generation ~= expected then return end
            if choice == 1 then invoke('check_source') elseif choice == 2 then invoke('generate') end
          end)
        elseif ({validate = true, save = true, save_and_apply_data = true, delivery_review = true, delivery_resume = true, reconcile = true, generate = true,
            apply_data = true, bundle = true, forward = true, observed_links = true, trial = true, train = true, runs = true, pending_runs = true,
            import_supervision = true, export_supervision = true, training_compare = true, source_to_node = true,
            capture_input = true, replay_input = true, discard_input = true, locate_failure = true, language = true})[command] then
          invoke(command)
        end
      end,
      select = function(id, requested_action)
        if not alive() or generation ~= expected or not find_node(id) then return end
        selected, selected_edge = id, nil
        if focus_root then focus_root = id end
        render(); details(requested_action)
      end,
      edge = function(index)
        if not alive() or generation ~= expected then return end
        local edge = document.edges[index]
        if not edge then return end
        selected, selected_edge = nil, edge
        if focus_root then focus_root = nil end
        render(); edge_details(edge)
      end,
      connect = function(source_id, target_id)
        local source, target = find_node(source_id), find_node(target_id)
        if not alive() or generation ~= expected or not source or not target then return end
        vim.ui.select(source.outputs, {prompt = tr('Choose output port', '选择输出端口', '選擇輸出連接埠')}, function(output)
          if not output or not alive() or generation ~= expected then return end
          vim.ui.select(target.inputs, {prompt = tr('Choose input port', '选择输入端口', '選擇輸入連接埠')}, function(input)
            if not input or not alive() or generation ~= expected then return end
            connect_ports({node = source_id, port = output, generation = expected}, target_id, input)
          end)
        end)
      end,
      connect_ports = function(source_id, output, target_id, input)
        if not alive() or generation ~= expected then return end
        connect_ports({node = source_id, port = output, generation = expected}, target_id, input)
      end,
    })
    if controller then
      plane_show_source = function(source_right)
        if controller.alive() then controller.show_source(source_right) else plane_show_source = nil end
      end
      plane_refresh = function(next_doc, next_positions, next_generation)
        if not controller.alive() then return false end
        expected = next_generation
        controller.refresh(next_doc, next_positions)
        return true
      end
      plane_status = function(info, focus)
        if not controller.alive() then plane_status = nil; return end
        controller.status(info, focus)
      end
    end
  end
  local function toggle_focus()
    if pending or dragging then
      message = tr('Finish or cancel the current connection first.', '请先完成或取消当前接线。', '請先完成或取消目前的接線。'); return render()
    end
    if focus_root then focus_root = nil
    elseif selected and find_node(selected) then focus_root = selected
    else message = tr('Select a node first.', '请先选择节点。', '請先選擇節點。'); return render() end
    message = nil; render()
    if selected_row then
      api.nvim_win_set_cursor(win, {selected_row, 0})
      api.nvim_win_call(win, function() vim.cmd('normal! zz') end)
    end
  end
  vim.keymap.set('n', 'a', add_node, {buffer = buf, nowait = true})
  vim.keymap.set('n', 'H', add_structure, {buffer = buf, nowait = true})
  vim.keymap.set('n', '/', find_nodes, {buffer = buf, nowait = true})
  vim.keymap.set('n', 'o', auto_layout, {buffer = buf, nowait = true})
  vim.keymap.set('n', 'G', manage_groups, {buffer = buf, nowait = true})
  vim.keymap.set('n', 'z', toggle_focus, {buffer = buf, nowait = true})
  vim.keymap.set('n', '<CR>', function()
    local row = api.nvim_win_get_cursor(win)[1]
    local id, edge = rows[row], edge_rows[row]
    local group_id = group_rows[row]
    if group_id and not pending then
      collapsed_groups[group_id] = not collapsed_groups[group_id] or nil
      if collapsed_groups[group_id] then selected, selected_edge = nil, nil end
      layout_changed(); render()
    elseif edge and not pending then
      selected, selected_edge = nil, edge; render(); edge_details(edge)
    elseif id then
      if pending then select_node(id) else selected, selected_edge = id, nil; render(); details() end
    else details() end
  end, {buffer = buf})
  vim.keymap.set('n', 'c', function() local node = selected and find_node(selected); if node then connect_start(node) end end, {buffer = buf, nowait = true})
  vim.keymap.set('n', 'x', remove_edge, {buffer = buf, nowait = true})
  vim.keymap.set('n', 'u', function() restore(false) end, {buffer = buf, nowait = true})
  vim.keymap.set('n', '<C-r>', function() restore(true) end, {buffer = buf, nowait = true})
  for _, key in ipairs({'v', 's', 'r', 'g'}) do vim.keymap.set('n', key, function() invoke(({v='validate', s='save', r='reconcile', g='generate'})[key]) end, {buffer = buf, nowait = true}) end
  vim.keymap.set('n', 'b', function() invoke('apply_data') end, {buffer = buf, nowait = true})
  vim.keymap.set('n', 't', function() invoke('trial') end, {buffer = buf, nowait = true})
  vim.keymap.set('n', 'Y', function() invoke('train') end, {buffer = buf, nowait = true})
  vim.keymap.set('n', 'R', function() invoke('runs') end, {buffer = buf, nowait = true})
  vim.keymap.set('n', 'Q', function() invoke('pending_runs') end, {buffer = buf, nowait = true})
  vim.keymap.set('n', 'L', link_supervision, {buffer = buf, nowait = true})
  vim.keymap.set('n', 'U', remove_supervision_reference, {buffer = buf, nowait = true})
  vim.keymap.set('n', 'V', function() invoke('import_supervision') end, {buffer = buf, nowait = true})
  vim.keymap.set('n', 'X', function() invoke('export_supervision') end, {buffer = buf, nowait = true})
  vim.keymap.set('n', 'T', configure_training, {buffer = buf, nowait = true})
  vim.keymap.set('n', 'P', open_plane, {buffer = buf, nowait = true})
  vim.keymap.set('n', 'p', function() invoke('forward') end, {buffer = buf, nowait = true})
  vim.keymap.set('n', 'd', function() invoke('locate_failure') end, {buffer = buf, nowait = true})
  vim.keymap.set('n', 'i', function() invoke('capture_input') end, {buffer = buf, nowait = true})
  vim.keymap.set('n', 'I', function() invoke('replay_input') end, {buffer = buf, nowait = true})
  vim.keymap.set('n', 'e', function() invoke('environment') end, {buffer = buf, nowait = true})
  local function port_at(row, column, direction)
    for _, hit in ipairs(port_hits[row] or {}) do
      if column >= hit.start and column < hit.finish and (not direction or hit.direction == direction) then return hit end
    end
  end
  local function drag_target(mode, row, column)
    if mode == 'wire' then return port_at(row, column, 'input') end
    if group_rows[row] then return {group = group_rows[row]} end
    return rows[row]
  end
  local scroll_timer, scroll_direction = (vim.uv or vim.loop).new_timer(), 0
  local function stop_drag_scroll()
    scroll_direction = 0
    if scroll_timer and not scroll_timer:is_closing() then scroll_timer:stop() end
  end
  local function scroll_drag(direction)
    if not scroll_timer or scroll_direction == direction then return end
    stop_drag_scroll()
    if direction == 0 then return end
    scroll_direction = direction
    scroll_timer:start(250, 100, vim.schedule_wrap(function()
      if not dragging or not alive() or scroll_direction == 0 then return stop_drag_scroll() end
      local view = api.nvim_win_call(win, vim.fn.winsaveview)
      local last_top = math.max(1, api.nvim_buf_line_count(buf) - api.nvim_win_get_height(win) + 1)
      local top = math.max(1, math.min(last_top, view.topline + scroll_direction))
      if top == view.topline then return end
      view.topline, view.lnum, view.col = top, math.min(api.nvim_buf_line_count(buf), top + dragging.viewport_row - 1), dragging.column
      api.nvim_win_call(win, function() vim.fn.winrestview(view) end)
      local target = drag_target(dragging.mode, view.lnum, dragging.column)
      if dragging.mode == 'wire' and type(target) ~= 'table' then target = nil end
      dragging.target = target
      render()
    end))
  end
  vim.keymap.set('n', '<LeftMouse>', function()
    local mouse = vim.fn.getmousepos()
    if mouse.winid ~= win then return end
    if group_rows[mouse.line] and not pending then
      local id = group_rows[mouse.line]
      collapsed_groups[id] = not collapsed_groups[id] or nil
      if collapsed_groups[id] then selected, selected_edge = nil, nil end
      layout_changed(); render(); return
    end
    if edge_rows[mouse.line] and not pending then
      selected, selected_edge = nil, edge_rows[mouse.line]
      api.nvim_win_set_cursor(win, {mouse.line, 0}); render()
      edge_details(selected_edge); return
    end
    if rows[mouse.line] then
      local id, connecting = rows[mouse.line], pending ~= nil
      local hit = port_at(mouse.line, mouse.column - 1)
      if connecting and hit and hit.direction == 'input' then connect_ports(pending, id, hit.port); return end
      api.nvim_win_set_cursor(win, {mouse.line, 0}); select_node(id)
      if not connecting and alive() then
        if hit and hit.direction == 'output' then
          dragging = {mode = 'wire', source = {node = id, port = hit.port, generation = generation}, generation = generation, moved = false}
        else dragging = {mode = 'node', source = id, target = id, generation = generation, moved = false} end
      end
      return
    end
    if not toolbar_rows[mouse.line] then return end
    local line = api.nvim_buf_get_lines(buf, mouse.line - 1, mouse.line, false)[1] or ''
    local marks = {}
    for pos, key in line:gmatch('()%[([^%]]+)%]') do marks[#marks + 1] = {pos = pos, key = key} end
    local action
    for index, item in ipairs(marks) do
      local next_pos = marks[index + 1] and marks[index + 1].pos or (#line + 1)
      if mouse.column >= item.pos and mouse.column < next_pos then action = item.key; break end
    end
    if action == 'a' then add_node()
    elseif action == 'H' then add_structure()
    elseif action == '/' then find_nodes()
    elseif action == 'o' then auto_layout()
    elseif action == 'G' then manage_groups()
    elseif action == 'z' then toggle_focus()
    elseif action == 'Enter' then details()
    elseif action == 'c' then local node = selected and find_node(selected); if node then connect_start(node) end
    elseif action == 'x' then remove_edge()
    elseif action == 'u' then restore(false)
    elseif action == 'Ctrl-R' then restore(true)
    elseif action == 'v' then invoke('validate')
    elseif action == 's' then invoke('save')
    elseif action == 'r' then invoke('reconcile')
    elseif action == 'b' then invoke('apply_data')
    elseif action == 't' then invoke('trial')
    elseif action == 'Y' then invoke('train')
    elseif action == 'R' then invoke('runs')
    elseif action == 'Q' then invoke('pending_runs')
    elseif action == 'T' then configure_training()
    elseif action == 'P' then open_plane()
    elseif action == 'p' then invoke('forward')
    elseif action == 'd' then invoke('locate_failure')
    elseif action == 'i' then invoke('capture_input')
    elseif action == 'I' then invoke('replay_input')
    elseif action == 'e' then invoke('environment')
    elseif action == 'g' then invoke('generate')
    elseif action == 'q' then api.nvim_win_close(win, true); if actions.back then actions.back() end end
  end, {buffer = buf, nowait = true})
  vim.keymap.set('n', '<LeftDrag>', function()
    if not dragging or not alive() then return end
    local mouse = vim.fn.getmousepos()
    if mouse.winid ~= win then stop_drag_scroll(); dragging.moved = true; dragging.target = nil; render(); return end
    local view = api.nvim_win_call(win, vim.fn.winsaveview)
    dragging.viewport_row = math.max(1, mouse.line - view.topline + 1)
    dragging.column = math.max(0, mouse.column - 1)
    local height = api.nvim_win_get_height(win)
    scroll_drag(dragging.viewport_row <= 1 and -1 or (dragging.viewport_row >= height and 1 or 0))
    local target = mouse.winid == win and drag_target(dragging.mode, mouse.line, mouse.column - 1) or nil
    if dragging.mode == 'wire' and type(target) ~= 'table' then target = nil end
    dragging.moved = true
    if not vim.deep_equal(dragging.target, target) then dragging.target = target; render() end
  end, {buffer = buf, nowait = true})
  vim.keymap.set('n', '<LeftRelease>', function()
    if not dragging or not alive() then return end
    local moving, mouse = dragging, vim.fn.getmousepos()
    stop_drag_scroll()
    local view = api.nvim_win_call(win, vim.fn.winsaveview)
    local line = moving.viewport_row and view.topline + moving.viewport_row - 1 or mouse.line
    local column = moving.column or mouse.column - 1
    local target = mouse.winid == win and drag_target(moving.mode, line, column) or nil
    if moving.mode == 'wire' and type(target) ~= 'table' then target = nil end
    dragging = nil
    if moving.mode == 'wire' then
      if moving.moved and target and generation == moving.generation then connect_ports(moving.source, target.node, target.port)
      elseif not moving.moved and generation == moving.generation then
        pending = moving.source; message = tr('Choose an input port to connect; Esc cancels.', '选择输入端口接线；Esc 取消。', '選擇輸入連接埠接線；Esc 取消。'); render()
      else message = tr('Connection cancelled; drop on an input port.', '接线已取消，请放到输入端口。', '接線已取消，請放到輸入連接埠。'); render() end
    elseif moving.moved and type(target) == 'table' and target.group and generation == moving.generation then
      local node, group = find_node(moving.source), nil
      for _, item in ipairs(document.groups or {}) do if item.id == target.group then group = item; break end end
      if node and group then
        local groups = vim.deepcopy(document.groups)
        for _, item in ipairs(groups) do
          for index = #item.nodes, 1, -1 do if item.nodes[index] == node.id then table.remove(item.nodes, index) end end
          if item.id == group.id then item.nodes[#item.nodes + 1] = node.id end
        end
        apply_edits({{action = 'set_groups', groups = groups}}, function(candidate)
          candidate.groups = vim.deepcopy(groups)
        end, function()
          message = tr('Node moved to group; connections are unchanged.', '节点已移入分组；连接未变化。', '節點已移入分組；連線未變更。'); render()
        end)
      end
    elseif moving.moved and type(target) == 'string' and generation == moving.generation then move_node(moving.source, target)
    elseif moving.moved then message = tr('Move cancelled; drop on a node.', '移动已取消，请放到节点上。', '移動已取消，請放到節點上。'); render()
    elseif mouse.winid == win and generation == moving.generation and selected == moving.source then
      render(); details()
    else render() end
  end, {buffer = buf, nowait = true})
  for key, offset in pairs({['['] = -1, [']'] = 1}) do
    vim.keymap.set('n', key, function()
      if not selected then return end
      local function membership(node_id)
        for _, group in ipairs(document.groups or {}) do if vim.tbl_contains(group.nodes, node_id) then return group.id end end
      end
      local current_group = membership(selected)
      for index, node_id in ipairs(layout_order) do
        if node_id == selected then
          local next_index = index + offset
          while layout_order[next_index] do
            local target = layout_order[next_index]
            if membership(target) == current_group then move_node(selected, target); break end
            next_index = next_index + offset
          end
          return
        end
      end
    end, {buffer = buf, nowait = true})
  end
  vim.keymap.set('n', '<Esc>', function()
    if dragging then stop_drag_scroll(); dragging = nil; message = tr('Move cancelled.', '移动已取消。', '移動已取消。'); render()
    elseif pending then pending = nil; message = nil; render()
    else api.nvim_win_close(win, true) end
  end, {buffer = buf, nowait = true})
  vim.keymap.set('n', 'q', function() api.nvim_win_close(win, true); if actions.back then actions.back() end end, {buffer = buf, nowait = true})
  local group = api.nvim_create_augroup('ModelMavenGraphRecovery' .. buf, {clear = true})
  local function relayout()
    if not alive() then return end
    local preview_width, preview_col = require('model_maven.model_preview').layout(110, actions.preview_boundary)
    local next_width = code_layout and math.max(1, math.floor((vim.o.columns - 5) / 2)) or preview_width
    local next_height = math.max(1, math.min(34, vim.o.lines - 5))
    local ok = pcall(api.nvim_win_set_config, win, {relative = 'editor', width = next_width, height = next_height,
      row = 1, col = code_layout and (source_on_right and 1 or math.max(0, vim.o.columns - next_width - 2)) or preview_col})
    if ok then render() end
  end
  api.nvim_create_autocmd('VimResized', {group = group, callback = relayout})
  api.nvim_create_autocmd('User', {group = group, pattern = 'ModelMavenPreviewLayout', callback = relayout})
  api.nvim_create_autocmd('BufWipeout', {group = group, buffer = buf, once = true, callback = function()
    stop_drag_scroll()
    if scroll_timer and not scroll_timer:is_closing() then scroll_timer:close() end
    if actions.checkpoint then actions.checkpoint(true) end
    pcall(api.nvim_del_augroup_by_id, group)
  end})
  api.nvim_create_autocmd('VimLeavePre', {group = group, callback = function()
    if actions.checkpoint then actions.checkpoint(true) end
  end})
  render()
  open_plane()
end
return M
