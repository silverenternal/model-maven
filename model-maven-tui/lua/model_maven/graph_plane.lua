-- Display-only grid for the graph editor. Positions never define data flow.
local M = {}

function M.open(document, positions, actions)
  local api, tr = vim.api, actions.tr
  local width, offset = require('model_maven.model_preview').layout(104, actions.preview_boundary)
  local height = math.min(36, vim.o.lines - 5)
  if width < 36 or height < 9 then return vim.notify(tr('Enlarge the terminal.', '请放大终端。', '請放大終端。'), vim.log.levels.WARN) end
  local columns, cell, max_column = math.max(1, math.min(4, math.floor((width - 2) / 24))), 24, 31
  local buf = api.nvim_create_buf(false, true)
  vim.b[buf].model_maven_graph_window = true
  vim.bo[buf].bufhidden, vim.bo[buf].swapfile = 'wipe', false
  local win = api.nvim_open_win(buf, true, {relative = 'editor', style = 'minimal', border = 'rounded', width = width,
    height = height, row = 1, col = offset})
  vim.wo[win].wrap = false
  local namespace = api.nvim_create_namespace('ModelMavenGraphPlane')
  local preview_namespace = api.nvim_create_namespace('ModelMavenGraphPlaneDrag')
  local placed, occupied, selected, dragging, first_row, wire_hits, nodes, focused, code_layout, source_on_right = {}, {}, nil, nil, 6, {}, {}, false, false, false
  local graph_counts, wire_cache, grid_cache, incoming, outgoing, node_labels = {}, nil, nil, {}, {}, {}
  local picked = nil
  local toolbar_hits = {}
  local function key(x, y) return x .. ':' .. y end
  local function alive() return api.nvim_win_is_valid(win) and api.nvim_win_get_buf(win) == buf end
  -- Keep saved logical columns even on a narrow terminal; horizontal scrolling
  -- is provided by Neovim, rather than rewriting stored positions on resize.
  local function update_layout(next_document, next_positions)
    wire_cache, grid_cache = nil, nil
    document, positions, placed, occupied, nodes = next_document, next_positions, {}, {}, {}
    node_labels = {}
    local definitions_by_id = {}
    for _, definition in ipairs(document.definitions or {}) do definitions_by_id[definition.id] = definition end
    for _, node in ipairs(document.nodes) do
      nodes[node.id] = node
      local kind = ({input = 'IN', output = 'OUT', call = 'OP', cast = 'CAST', clone = 'COPY', mean_dynamic = 'MEAN', assert_flat = 'CHECK', assert_batch = 'CHECK', after = 'WAIT', add = 'ADD', concat = 'CAT'})[node.kind] or '?'
      local definition = node.definition and definitions_by_id[node.definition]
      if node.kind == 'call' and definition and type(definition.factory) == 'string' then
        local name = definition.factory:match('([A-Za-z_][A-Za-z0-9_]*)$')
        if name then kind = #name > 8 and (name:sub(1, 7) .. '~') or name end
      end
      local available = 15 - #kind
      local plane_id = node.id:find('[\128-\255]') and ('u:' .. vim.fn.sha256(node.id):sub(1, 6)) or node.id
      local hint
      if node.kind == 'call' and definition and type(definition.params) == 'table' then
        local names = {}
        for name, value in pairs(definition.params) do
          if type(name) == 'string' and name:match('^[%w_]+$') and (type(value) == 'number' or type(value) == 'boolean') then
            names[#names + 1] = name
          end
        end
        table.sort(names)
        if names[1] then
          local text = names[1] .. '=' .. tostring(definition.params[names[1]])
          local shortened = #text > 8 and (text:sub(1, 7) .. '~') or text
          if available - #shortened - 1 >= 3 then
            hint = shortened
            available = available - #hint - 1
          end
        end
      end
      local shown = #plane_id > available and (plane_id:sub(1, available - 1) .. '~') or plane_id
      node_labels[node.id] = {caption = shown .. ' ' .. kind .. (hint and (' ' .. hint) or '')}
    end
    incoming, outgoing = {}, {}
    for _, edge in ipairs(document.edges or {}) do
      incoming[edge.to.node] = incoming[edge.to.node] or {}
      outgoing[edge.from.node] = outgoing[edge.from.node] or {}
      incoming[edge.to.node][#incoming[edge.to.node] + 1] = edge
      outgoing[edge.from.node][#outgoing[edge.from.node] + 1] = edge
    end
    if selected and not nodes[selected] then selected, focused = nil, false end
    for _, node in ipairs(document.nodes) do
      local point = positions[node.id]
      if type(point) == 'table' and point.x <= max_column and point.y <= 511 and not occupied[key(point.x, point.y)] then
        placed[node.id] = {x = point.x, y = point.y}; occupied[key(point.x, point.y)] = node.id
      end
    end
    local next_free = 0
    for _, node in ipairs(document.nodes) do
      if not placed[node.id] then
        while occupied[key(next_free % columns, math.floor(next_free / columns))] do next_free = next_free + 1 end
        local point = {x = next_free % columns, y = math.floor(next_free / columns)}
        placed[node.id], occupied[key(point.x, point.y)] = point, node.id
        next_free = next_free + 1
      end
    end
    local connected, counts = {}, {inputs = 0, outputs = 0, calls = 0, missing = 0}
    for _, edge in ipairs(document.edges or {}) do connected[edge.to.node .. '\0' .. edge.to.port] = true end
    for _, node in ipairs(document.nodes) do
      if node.kind == 'input' then counts.inputs = counts.inputs + 1
      elseif node.kind == 'output' then counts.outputs = counts.outputs + 1
      elseif node.kind == 'call' then counts.calls = counts.calls + 1 end
      for _, port in ipairs(node.inputs or {}) do
        if not connected[node.id .. '\0' .. port] then counts.missing = counts.missing + 1 end
      end
    end
    graph_counts = counts
  end
  update_layout(document, positions)
  local function toolbar()
    local options = {
      {tr('Structure', '结构', '結構'), 'structure'},
      {tr('+ Operator', '+ 算子', '+ 運算子'), 'add_operator'},
      {tr('Data', '数据', '資料'), 'choose_data_node'},
      {tr('Check', '检查', '檢查'), 'validate'},
      {tr('Run', '运行', '執行'), 'forward'},
      {tr('Code', '源码', '原始碼'), 'source'},
    }
    local more = '[' .. tr('More', '更多', '更多') .. ' m]'
    local line = ' '
    toolbar_hits = {}
    local limit = math.max(1, api.nvim_win_get_width(win))
    for _, item in ipairs(options) do
      local label = '[' .. item[1] .. '] '
      if vim.fn.strdisplaywidth(line .. label .. more) > limit then break end
      local start = #line + 1
      line = line .. label
      toolbar_hits[#toolbar_hits + 1] = {first = start, last = #line, command = item[2]}
    end
    local start = #line + 1
    line = line .. more
    toolbar_hits[#toolbar_hits + 1] = {first = start, last = #line, command = 'menu'}
    return line
  end
  local function next_step()
    local inputs, outputs, calls, missing = graph_counts.inputs, graph_counts.outputs, graph_counts.calls, graph_counts.missing
    if #document.nodes == 0 then return tr('Start: choose a basic structure [H] or add a node [a]', '开始：选择基础结构 [H] 或添加节点 [a]', '開始：選擇基礎結構 [H] 或新增節點 [a]'), 'structure' end
    if inputs == 0 or outputs == 0 then return tr('Next: add input and output nodes [a]', '下一步：添加输入和输出节点 [a]', '下一步：新增輸入和輸出節點 [a]'), 'add' end
    if calls == 0 then return tr('Next: add an operator [o], or plan this graph [v]', '下一步：添加算子 [o]，或规划当前图 [v]', '下一步：新增運算子 [o]，或規劃目前的圖 [v]'), 'add_operator' end
    if missing > 0 then return tr('Connect an input port · unconnected: ', '连接输入端口 · 未连接：', '連接輸入連接埠 · 未連接：') .. missing .. tr(' · plan [v]', ' · 检查 [v]', ' · 檢查 [v]'), 'connect_missing' end
    if not document.data then return tr('Choose a node to connect data', '选择节点连接数据', '選擇節點連接資料'), 'choose_data_node' end
    if actions.is_saved and not actions.is_saved(document) then
      local status = document.data and actions.data_status and actions.data_status(document) or 'unknown'
      if status == 'matching' or status == 'different' or status == 'session_override' then
        return tr('Save graph, then review project data [S]', '保存图并审核项目数据 [S]', '儲存圖並審核專案資料 [S]'), 'save_and_apply_data'
      end
      return tr('Next: review and save graph [s]', '下一步：审核并保存图 [s]', '下一步：檢視並儲存圖 [s]'), 'save'
    end
    local data_status = actions.data_status and actions.data_status(document) or 'unknown'
    if data_status == 'project_missing' then return tr('Generate source and set up the project [g]', '生成源码并初始化项目 [g]', '產生原始碼並初始化專案 [g]'), 'generate' end
    if actions.source_status and actions.source_status(document) ~= 'last_checked' then
      return tr('Check or generate the project model source', '核对或生成项目模型源码', '核對或產生專案模型原始碼'), 'source_check'
    end
    if data_status == 'different' then return tr('Graph data differs from project input · review [b]', '图数据与项目输入不同 · 审核 [b]', '圖資料與專案輸入不同 · 檢視 [b]'), 'apply_data' end
    if data_status == 'session_override' then return tr('Session input overrides graph data · review [b]', '会话输入覆盖图数据 · 审核 [b]', '工作階段輸入覆蓋圖資料 · 檢視 [b]'), 'apply_data' end
    if data_status == 'unknown' then return tr('Project input status is unknown · review [b]', '项目输入状态未知 · 审核 [b]', '專案輸入狀態未知 · 檢視 [b]'), 'apply_data' end
    if #(document.objectives or {}) == 0 then return tr('For training, add a loss · or plan [v]', '训练可添加损失 · 或规划 [v]', '訓練可新增損失 · 或規劃 [v]'), 'choose_objective_node' end
    return tr('Next: plan [v], generate [g], forward [p], trial [t] or train [Y]', '下一步：规划 [v]、生成 [g]、前向 [p]、试跑 [t] 或训练 [Y]', '下一步：規劃 [v]、產生 [g]、前向 [p]、試跑 [t] 或訓練 [Y]'), 'validate'
  end
  local function render(message)
    if not alive() then return end
    local view = api.nvim_win_call(win, vim.fn.winsaveview)
    local selected_summary, selected_detail
    if selected and nodes[selected] then
      local node = nodes[selected]
      local factory, definition
      for _, item in ipairs(document.definitions or {}) do
        if item.id == node.definition then definition, factory = item, item.factory; break end
      end
      local shared_count = 0
      if definition then
        for _, call in ipairs(document.nodes) do if call.definition == definition.id then shared_count = shared_count + 1 end end
      end
      local groups, group_path = {}, {}
      for _, group in ipairs(document.groups or {}) do groups[group.id] = group end
      local membership
      for _, group in ipairs(document.groups or {}) do
        if vim.tbl_contains(group.nodes or {}, node.id) then membership = group; break end
      end
      local seen = {}
      while membership and not seen[membership.id] and #group_path < 32 do
        seen[membership.id] = true
        table.insert(group_path, 1, membership.id)
        membership = membership.parent and groups[membership.parent] or nil
      end
      selected_summary = tr('Selected: ', '已选：', '已選取：') .. selected .. ' · ' .. node.kind
        .. (node.kind == 'output' and node.output_name and (' · ' .. tr('output ', '输出 ', '輸出 ') .. node.output_name) or '')
        .. (shared_count > 1 and (tr(' · shared ×', ' · 共享 ×', ' · 共用 ×') .. shared_count) or '')
        .. ' · ' .. tr('ports ', '端口 ', '連接埠 ') .. #(node.inputs or {}) .. ' → ' .. #(node.outputs or {})
        .. (factory and (' · ' .. vim.fn.strcharpart(factory:gsub('[%c]', ' '), 0, 120)) or '')
        .. (focused and tr(' · only adjacent wires shown', ' · 仅显示相邻连接', ' · 僅顯示相鄰連線') or '')
      if #group_path > 0 then
        selected_summary = selected_summary .. tr(' · display group: ', ' · 显示分组：', ' · 顯示分組：') .. table.concat(group_path, ' / ')
      end
      local details = {}
      if node.kind == 'call' and definition then
        local names = vim.tbl_keys(definition.params or {}); table.sort(names)
        for index, name in ipairs(names) do
          if index > 3 then details[#details + 1] = '…'; break end
          local ok, value = pcall(vim.json.encode, definition.params[name])
          details[#details + 1] = name .. '=' .. (ok and vim.fn.strcharpart(value:gsub('[%c]', ' '), 0, 40) or '?')
        end
        selected_detail = tr('Parameters: ', '参数：', '參數：') .. (#details > 0 and table.concat(details, ', ') or tr('none declared', '未声明', '未宣告'))
          .. tr(' · click for details', ' · 点击查看详情', ' · 點選查看詳情')
      elseif node.kind == 'output' then
        for _, objective in ipairs(document.objectives or {}) do
          if objective.output == node.id then
            details[#details + 1] = objective.loss .. ' → ' .. objective.target .. ' × ' .. tostring(objective.weight)
            if #details >= 3 then break end
          end
        end
        selected_detail = tr('Losses: ', '损失：', '損失：') .. (#details > 0 and table.concat(details, ', ') or tr('not declared', '未声明', '未宣告'))
          .. tr(' · click to configure', ' · 点击配置', ' · 點選設定')
      elseif node.kind == 'input' then
        selected_detail = tr('Data: ', '数据：', '資料：') .. (document.data and tostring(document.data.factory):gsub('[%c]', ' ')
          or tr('not declared', '未声明', '未宣告')) .. tr(' · click to connect', ' · 点击连接', ' · 點選連接')
        if document.data and actions.data_status then
          local status = actions.data_status(document)
          local text = ({matching = tr('project input matches', '项目输入匹配', '專案輸入相符'),
            different = tr('project input differs', '项目输入不同', '專案輸入不同'),
            session_override = tr('session input overrides', '会话输入覆盖', '工作階段輸入覆蓋'),
            project_missing = tr('project not configured', '项目未初始化', '專案未初始化'),
            unknown = tr('project input status unknown', '项目输入状态未知', '專案輸入狀態未知')})[status]
          if text then selected_detail = selected_detail .. ' · ' .. text .. tr(' (last checked)', '（上次检查）', '（上次檢查）') end
        end
      elseif node.kind == 'concat' then
        selected_detail = tr('Axis: ', '拼接轴：', '串接軸：') .. tostring(node.axis ~= nil and node.axis or '?') .. tr(' · click to edit', ' · 点击编辑', ' · 點選編輯')
      end
      if node.kind == 'input' or node.kind == 'output' then
        local count = 0
        for _, rules in pairs(node.draft_rules or {}) do count = count + #rules end
        if count > 0 then
          selected_detail = (selected_detail or '') .. tr(' · draft rules (binding unknown): ', ' · 草稿规则（绑定状态未知）：', ' · 草稿規則（繫結狀態未知）：') .. count
        end
        if document.supervision then
          selected_detail = (selected_detail or '') .. tr(' · linked rules: binding unknown', ' · 关联规则：绑定状态未知', ' · 連結規則：繫結狀態未知')
        end
      end
      if node.kind == 'call' then
        selected_detail = (selected_detail or '') .. tr(' · [M] reorder', ' · [M] 调序', ' · [M] 調序')
      end
      if #group_path > 0 then
        selected_detail = (selected_detail or '') .. tr(' · groups are layout only [G]', ' · 分组只影响布局 [G]', ' · 分組只影響佈局 [G]')
      end
    end
    local next_text = picked and (tr('Moving ', '正在移动 ', '正在移動 ') .. picked.id
      .. tr(' · arrows/Tab choose target · Enter place · Esc cancel', ' · 方向键/Tab 选择目标 · Enter 放置 · Esc 取消', ' · 方向鍵/Tab 選擇目標 · Enter 放置 · Esc 取消')) or next_step()
    local unicode_names = false
    for _, node in ipairs(document.nodes) do if node.id:find('[\128-\255]') then unicode_names = true; break end end
    local lines = {tr(' MODEL GRAPH · IN input / operator type / ADD sum / CAT concat / OUT output',
        ' 模型图 · IN 输入 / 算子类型 / ADD 相加 / CAT 拼接 / OUT 输出',
        ' 模型圖 · IN 輸入 / 運算子類型 / ADD 相加 / CAT 串接 / OUT 輸出'),
      toolbar(),
      ' [n] ' .. next_text,
      message or (selected_summary
        or tr('Drag o to i to connect; click wires to inspect. Intersections may represent multiple connections.', '从 o 拖到 i 接线；点击线条查看连接。交叉处可能有多条连接。', '從 o 拖到 i 接線；點選線條檢視連線。交叉處可能有多條連線。')),
      selected_detail or (unicode_names and tr('Unicode names use u: tags on the plane; select a block to see its full name.',
        '平面中 Unicode 名称以 u: 标签显示；选中后查看全名。', '平面中 Unicode 名稱以 u: 標籤顯示；選取後查看全名。')
        or tr('Tab: select · Enter: details · PgUp/PgDn: scroll · m: actions',
          'Tab 选择 · Enter 详情 · PgUp/PgDn 翻页 · m 操作', 'Tab 選取 · Enter 詳情 · PgUp/PgDn 翻頁 · m 操作'))}
    local max_y, max_x = 0, columns - 1
    for _, point in pairs(placed) do max_y, max_x = math.max(max_y, point.y), math.max(max_x, point.x) end
    max_x = math.min(max_column, max_x + 1)
    local cache_key = focused and (selected or '') or false
    local wires, truncated
    if wire_cache and wire_cache.focus == cache_key then
      wires, wire_hits, truncated = wire_cache.wires, wire_cache.hits, wire_cache.truncated
    else
    local drawn, budget
    wires, wire_hits, truncated, drawn, budget = {}, {}, false, 0, 100000
    local function put(row, col, char, edge_index)
      if budget <= 0 then truncated = true; return end
      budget = budget - 1
      wires[row] = wires[row] or {}
      local old = wires[row][col]
      local id = row .. ':' .. col
      wire_hits[id] = wire_hits[id] or {}
      local hits = wire_hits[id]
      if hits[#hits] ~= edge_index then hits[#hits + 1] = edge_index end
      if #hits > 1 then wires[row][col] = '*'
      elseif char == '^' or char == 'v' or char == '+' then wires[row][col] = char
      else wires[row][col] = old and old ~= char and '+' or char end
    end
    for index, edge in ipairs(document.edges or {}) do
      local source, target = placed[edge.from.node], placed[edge.to.node]
      if source and target and (not focused or edge.from.node == selected or edge.to.node == selected) then
        if drawn >= 256 or budget <= 0 then truncated = true; break end
        drawn = drawn + 1
        local sx, tx = source.x * cell, target.x * cell
        local sy, ty = first_row + source.y * 2, first_row + target.y * 2
        local cost = math.abs(sx - tx) + 1 + math.abs(sy + 1 - ty) + 1 + 2
        if cost > budget then truncated = true; break end
        for col = math.min(sx, tx), math.max(sx, tx) do put(sy + 1, col, '-', index) end
        for row = math.min(sy + 1, ty), math.max(sy + 1, ty) do put(row, tx, '|', index) end
        put(sy + 1, sx, '+', index); put(ty, tx, ty <= sy and '^' or 'v', index)
      end
    end
    wire_cache = {focus = cache_key, wires = wires, hits = wire_hits, truncated = truncated}
    end
    if truncated then lines[1] = tr(' ! Some wires hidden · [q] connection view',
      ' ! 部分连线隐藏 · [q] 连接视图', ' ! 部分連線隱藏 · [q] 連線檢視') end
    if not grid_cache or grid_cache.focus ~= cache_key then
    -- Keep an unselected grid: changing focus only patches one marker instead
    -- of rebuilding every empty cell and re-pruning every wire hit.
    local visible_hits = {}
    for position, hits in pairs(wire_hits) do visible_hits[position] = hits end
    local node_marks, grid_lines = {}, {}
    local blank_row = string.rep(' ', cell * (max_x + 1))
    for y = 0, max_y + 1 do
      local row = {}
      for x = 0, max_x do
        local id = occupied[key(x, y)]
        local node = id and nodes[id]
        local label
        if id then
          -- Cached on document/layout changes; selection redraws must not hash
          -- every Unicode node again. Full names remain in the detail line.
          local token = node_labels[id]
          label = ' ' .. (#(node.inputs or {}) > 0 and 'i' or ' ') .. ' [' .. token.caption .. ']'
          node_marks[#node_marks + 1] = {row = first_row + y * 2, start_col = x * cell + 2,
            end_col = x * cell + #label, id = id, kind = node.kind, input = #(node.inputs or {}) > 0,
            output = #(node.outputs or {}) > 0, x = x}
        else label = '  .' end
        label = string.format('%-22s', label:sub(1, 21)) .. (node and #(node.outputs or {}) > 0 and 'o' or ' ') .. ' '
        row[#row + 1] = label
      end
      for _, text in ipairs({table.concat(row), blank_row}) do
        local row_number = first_row + #grid_lines
        local values = wires[row_number]
        if values then
          local chars = {}; for col = 1, #text do chars[col] = text:sub(col, col) end
          for col, char in pairs(values) do
            if col >= 0 and col < #text and chars[col + 1] == ' ' then chars[col + 1] = char
            else visible_hits[row_number .. ':' .. col] = nil end
          end
          text = table.concat(chars)
        end
        grid_lines[#grid_lines + 1] = text
      end
    end
    grid_cache = {focus = cache_key, lines = grid_lines, marks = node_marks, hits = visible_hits}
    end
    wire_hits = grid_cache.hits
    local node_marks = grid_cache.marks
    for _, line in ipairs(grid_cache.lines) do lines[#lines + 1] = line end
    if selected and placed[selected] then
      local point = placed[selected]
      local row, col = first_row + point.y * 2, point.x * cell + 3
      local line = lines[row]
      if line then
        lines[row] = line:sub(1, col - 1) .. '>' .. line:sub(col + 1)
        local hit_key = row .. ':' .. (col - 1)
        if wire_hits[hit_key] then
          wire_hits = vim.tbl_extend('force', {}, wire_hits)
          wire_hits[hit_key] = nil
        end
      end
    end
    vim.bo[buf].modifiable = true; api.nvim_buf_set_lines(buf, 0, -1, false, lines); vim.bo[buf].modifiable = false
    api.nvim_buf_clear_namespace(buf, namespace, 0, -1)
    api.nvim_buf_set_extmark(buf, namespace, 0, 0, {end_col = #lines[1], hl_group = 'Title'})
    api.nvim_buf_set_extmark(buf, namespace, 1, 0, {end_col = #lines[2], hl_group = 'Comment'})
    if selected then api.nvim_buf_set_extmark(buf, namespace, 4, 0, {end_col = #lines[5], hl_group = 'NormalFloat'}) end
    for _, mark in ipairs(node_marks) do
      local group = mark.id == selected and 'Visual' or (mark.kind == 'input' or mark.kind == 'output') and 'Identifier'
        or mark.kind == 'call' and 'Type' or 'Statement'
      api.nvim_buf_set_extmark(buf, namespace, mark.row - 1, mark.start_col, {end_col = mark.end_col, hl_group = group})
      if mark.input then api.nvim_buf_set_extmark(buf, namespace, mark.row - 1, mark.x * cell + 1,
        {end_col = mark.x * cell + 2, hl_group = 'DiagnosticInfo'}) end
      if mark.output then api.nvim_buf_set_extmark(buf, namespace, mark.row - 1, mark.x * cell + 22,
        {end_col = mark.x * cell + 23, hl_group = 'DiagnosticOk'}) end
    end
    view.lnum = math.min(view.lnum, #lines)
    api.nvim_win_call(win, function() vim.fn.winrestview(view) end)
  end
  local function cell_at(row, column)
    local y = math.floor((row - first_row) / 2)
    local x = math.floor(math.max(0, column) / cell)
    if row < first_row or x > max_column or y < 0 or y > 511 then return nil end
    return x, y
  end
  local function reorder_candidate(source, target)
    local function simple(id)
      local node, ins, outs = nodes[id], incoming[id] or {}, outgoing[id] or {}
      return node and node.kind == 'call' and #(node.inputs or {}) == 1 and #(node.outputs or {}) == 1
        and #ins == 1 and #outs == 1 and ins[1].to.port == node.inputs[1] and outs[1].from.port == node.outputs[1]
    end
    if not simple(source) or not simple(target) then return false end
    local first, seen = source, {}
    while simple(first) and not seen[first] do
      seen[first] = true
      local previous = incoming[first][1].from.node
      if not simple(previous) then break end
      first = previous
    end
    seen = {}
    while simple(first) and not seen[first] do
      if first == target then return true end
      seen[first] = true; first = outgoing[first][1].to.node
    end
    return false
  end
  local function drag_preview(row, column, outside)
    if not alive() then return end
    api.nvim_buf_clear_namespace(buf, preview_namespace, 0, -1)
    if not dragging or not dragging.moved then return end
    local hint, x, y, target
    dragging.valid_grid = false
    if outside then
      hint = tr('Outside canvas: release cancels.', '画布外：释放将取消。', '畫布外：放開將取消。')
    else
      x, y = cell_at(row, column)
      target = x and occupied[key(x, y)] or nil
      local target_row = y and first_row + y * 2 or nil
      local within = target_row and target_row <= api.nvim_buf_line_count(buf)
        and x * cell < #(api.nvim_buf_get_lines(buf, target_row - 1, target_row, false)[1] or '')
      dragging.valid_grid = within == true
      if not within then
        hint = tr('Outside the available grid: release cancels.', '超出可用网格：释放将取消。', '超出可用網格：放開將取消。')
      elseif dragging.mode == 'wire' then
        if target and target ~= dragging.id and (row - first_row) % 2 == 0
            and column % cell == 1 and #(nodes[target].inputs or {}) > 0 then
          hint = tr('Connection candidate: ', '接线候选：', '接線候選：') .. target .. tr(' · ports and compatibility checked after release', ' · 释放后选择端口并检查兼容性', ' · 放開後選擇連接埠並檢查相容性')
        else
          hint = tr('Drop on an input i to choose ports; otherwise cancels.', '放到输入 i 上选择端口；其他位置取消。', '放到輸入 i 上選擇連接埠；其他位置取消。')
        end
      elseif target and target ~= dragging.id then
        if reorder_candidate(dragging.id, target) then
          local after = actions.reorder_after and column % cell >= math.floor(cell / 2)
          hint = (after and tr('Release to review moving after ', '释放后审核移至其后：', '放開後檢視移至其後：')
            or tr('Release to review moving before ', '释放后审核移至其前：', '放開後檢視移至其前：')) .. target
        else
          hint = tr('Different or branched paths: use wire editing; this reorder is unavailable.',
            '节点不在同一无分支链：请编辑连线，此处不能重排。', '節點不在同一無分支鏈：請編輯連線，此處不能重排。')
        end
      elseif target == dragging.id then
        hint = tr('Current position: no move.', '当前位置：不会移动。', '目前位置：不會移動。')
      else
        hint = tr('Empty position: release moves display only.', '空位置：释放只移动显示位置。', '空位置：放開只移動顯示位置。')
      end
      if within then
        local after = dragging.mode == 'node' and target and target ~= dragging.id and actions.reorder_after
          and column % cell >= math.floor(cell / 2)
        api.nvim_buf_set_extmark(buf, preview_namespace, target_row - 1, x * cell + (after and cell - 1 or 0), {
          virt_text = {{dragging.mode == 'wire' and '*' or (after and '>' or '<'), 'Search'}}, virt_text_pos = 'overlay', priority = 200,
        })
      end
    end
    if hint then
      api.nvim_buf_set_extmark(buf, preview_namespace, 3, 0, {
        virt_text = {{'[ * ] ' .. hint, 'Search'}}, virt_text_pos = 'overlay', priority = 200,
      })
    end
  end
  local function move(id, x, y, follow, drop_after)
    if not id or x < 0 or x > max_column or y < 0 or y > 511 then return end
    local current = placed[id]
    local existing = occupied[key(x, y)]
    if existing and existing ~= id then
      if not reorder_candidate(id, existing) then
        return render(tr('Cannot reorder across a branch or disconnected paths. Select a wire to edit connections.',
          '不能跨分支或不相连的路径重排。请选择连线进行编辑。', '不能跨分支或不相連的路徑重排。請選擇連線進行編輯。'))
      end
      local options = {tr('Move before this node in execution order', '在执行顺序中移到此节点之前', '在執行順序中移至此節點之前')}
      if actions.reorder_after then options[#options + 1] = tr('Move after this node in execution order', '在执行顺序中移到此节点之后', '在執行順序中移至此節點之後') end
      options[#options + 1] = tr('Cancel', '取消', '取消')
      local function reorder(after)
        actions.reorder(id, existing, after, vim.deepcopy(placed), function(next_doc, next_positions, error)
          if not alive() then return end
          if not next_doc then if error then render(error) end; return end
          update_layout(next_doc, next_positions)
          render(tr('Execution order changed; validate and run before relying on it.', '执行顺序已改变；使用前请规划并运行验证。', '執行順序已變更；使用前請規劃並執行驗證。'))
        end)
      end
      if drop_after ~= nil then reorder(drop_after); return end
      vim.ui.select(options, {
        prompt = tr('Occupied position: reorder a supported unbranched chain?', '目标位置已有节点：是否重排受支持的无分支链？', '目標位置已有節點：是否重排受支援的無分支鏈？'),
      }, function(_, choice)
        if not alive() or not choice or choice == #options then return end
        reorder(choice == 2)
      end)
      return
    end
    if current.x == x and current.y == y then return end
    wire_cache, grid_cache = nil, nil
    local old_x, old_y = current.x, current.y
    occupied[key(old_x, old_y)] = nil; current.x, current.y = x, y; occupied[key(x, y)] = id
    if not actions.move(vim.deepcopy(placed)) then
      occupied[key(x, y)] = nil; current.x, current.y = old_x, old_y; occupied[key(old_x, old_y)] = id
      return render(tr('Graph changed; reopen this view.', '图已变化，请重新打开此视图。', '圖已變更，請重新開啟此檢視。'))
    end
    render(tr('Display position changed; connections are unchanged.', '显示位置已更改，连接未变化。', '顯示位置已變更，連線未變更。'))
    if follow and alive() then api.nvim_win_set_cursor(win, {first_row + y * 2, x * cell + 3}) end
  end
  local function arrange()
    if dragging then return render(tr('Finish or cancel the gesture first.', '请先完成或取消当前操作。', '請先完成或取消目前操作。')) end
    local degree, successors, ranks, ready, groups = {}, {}, {}, {}, {}
    for _, node in ipairs(document.nodes) do
      if degree[node.id] ~= nil then return render(tr('Duplicate node IDs; fix the graph first.', '节点 ID 重复，请先修正图。', '節點 ID 重複，請先修正圖。')) end
      degree[node.id], successors[node.id], ranks[node.id] = 0, {}, 0
    end
    for _, edge in ipairs(document.edges or {}) do
      if degree[edge.from.node] == nil or degree[edge.to.node] == nil then
        return render(tr('A connection references a missing node.', '连接引用了不存在的节点。', '連線參照了不存在的節點。'))
      end
      degree[edge.to.node] = degree[edge.to.node] + 1
      successors[edge.from.node][#successors[edge.from.node] + 1] = edge.to.node
    end
    for _, node in ipairs(document.nodes) do if degree[node.id] == 0 then ready[#ready + 1] = node.id end end
    local processed, at = 0, 1
    while at <= #ready do
      local id = ready[at]; at, processed = at + 1, processed + 1
      for _, target in ipairs(successors[id]) do
        ranks[target] = math.max(ranks[target], ranks[id] + 1)
        degree[target] = degree[target] - 1
        if degree[target] == 0 then ready[#ready + 1] = target end
      end
    end
    if processed ~= #document.nodes then return render(tr('Cannot arrange a cyclic graph.', '无法排列包含循环的图。', '無法排列包含循環的圖。')) end
    local max_rank = 0
    for _, node in ipairs(document.nodes) do
      local rank = ranks[node.id]; max_rank = math.max(max_rank, rank)
      groups[rank] = groups[rank] or {}; groups[rank][#groups[rank] + 1] = node.id
    end
    local proposed, row = {}, 0
    for rank = 0, max_rank do
      local group = groups[rank] or {}
      for index, id in ipairs(group) do
        proposed[id] = {x = (index - 1) % columns, y = row + math.floor((index - 1) / columns)}
      end
      row = row + math.ceil(#group / columns)
    end
    if row > 512 then return render(tr('Graph is too large for this grid.', '图过大，超出此网格范围。', '圖過大，超出此網格範圍。')) end
    vim.ui.select({tr('Arrange by dependencies', '按依赖排列', '按相依關係排列'), tr('Cancel', '取消', '取消')}, {
      prompt = tr('Replace display positions? You can undo this in the graph view.',
        '替换显示位置？可在图视图撤销。', '取代顯示位置？可在圖檢視復原。'),
    }, function(_, choice)
      if choice ~= 1 or not alive() or dragging then return end
      if not actions.move(vim.deepcopy(proposed)) then return render(tr('Graph changed; reopen this view.', '图已变化，请重新打开此视图。', '圖已變更，請重新開啟此檢視。')) end
      wire_cache, grid_cache = nil, nil
      placed, occupied = proposed, {}
      for id, point in pairs(placed) do occupied[key(point.x, point.y)] = id end
      render(tr('Nodes arranged by dependency; connections are unchanged.', '节点已按依赖排列；连接未变化。', '節點已依相依關係排列；連線未變更。'))
    end)
  end
  local timer, scroll_y, scroll_x = (vim.uv or vim.loop).new_timer(), 0, 0
  local function stop_scroll()
    scroll_y, scroll_x = 0, 0
    if timer and not timer:is_closing() then timer:stop() end
  end
  local function scroll_drag(direction_y, direction_x)
    if not timer or (scroll_y == direction_y and scroll_x == direction_x) then return end
    stop_scroll()
    if direction_y == 0 and direction_x == 0 then return end
    scroll_y, scroll_x = direction_y, direction_x
    timer:start(250, 100, vim.schedule_wrap(function()
      if not dragging or not alive() or (scroll_y == 0 and scroll_x == 0) then return stop_scroll() end
      local view = api.nvim_win_call(win, vim.fn.winsaveview)
      local last_top = math.max(1, api.nvim_buf_line_count(buf) - api.nvim_win_get_height(win) + 1)
      local top = math.max(1, math.min(last_top, view.topline + scroll_y))
      local furthest = columns - 1; for _, point in pairs(placed) do furthest = math.max(furthest, point.x) end
      local grid_width = (math.min(max_column, furthest + 1) + 1) * cell
      local left = math.max(0, math.min(math.max(0, grid_width - api.nvim_win_get_width(win)), (view.leftcol or 0) + scroll_x * 2))
      if top == view.topline and left == (view.leftcol or 0) then return end
      dragging.column = math.max(0, dragging.column + left - (view.leftcol or 0))
      view.topline, view.lnum = top, math.min(api.nvim_buf_line_count(buf), top + dragging.viewport_row - 1)
      view.leftcol, view.col = left, dragging.column
      api.nvim_win_call(win, function() vim.fn.winrestview(view) end)
      drag_preview(view.topline + dragging.viewport_row - 1, dragging.column, false)
    end))
  end
  api.nvim_create_autocmd('BufWipeout', {buffer = buf, once = true, callback = function()
    stop_scroll(); if timer and not timer:is_closing() then timer:close() end
  end})
  local resize_group = api.nvim_create_augroup('ModelMavenPlane' .. buf, {clear = true})
  local function cancel_gesture()
    if not dragging then return false end
    dragging = nil
    stop_scroll()
    if alive() then api.nvim_buf_clear_namespace(buf, preview_namespace, 0, -1) end
    return true
  end
  api.nvim_create_autocmd('FocusLost', {group = resize_group, callback = function()
    if not alive() then return end
    local cancelled = cancel_gesture() or picked ~= nil
    picked = nil
    if cancelled then render(tr('Gesture cancelled on focus loss.', '窗口失焦，操作已取消。', '視窗失焦，操作已取消。')) end
  end})
  local function relayout()
    if not alive() then pcall(api.nvim_del_augroup_by_id, resize_group); return end
    local cancelled = cancel_gesture() or picked ~= nil
    picked = nil
    local preview_width, preview_col = require('model_maven.model_preview').layout(104, actions.preview_boundary)
    local next_width = code_layout and math.max(1, math.floor((vim.o.columns - 5) / 2)) or preview_width
    local next_height = math.max(1, math.min(36, vim.o.lines - 5))
    if next_height < 4 then return end
    api.nvim_win_set_config(win, {relative = 'editor', row = 1, col = code_layout and (source_on_right and 1 or math.max(0, vim.o.columns - next_width - 2)) or preview_col,
      width = next_width, height = next_height})
    render(cancelled and tr('Gesture cancelled on resize.', '窗口大小变化，操作已取消。', '視窗大小變更，操作已取消。') or nil)
  end
  api.nvim_create_autocmd('VimResized', {group = resize_group, callback = relayout})
  api.nvim_create_autocmd('User', {group = resize_group, pattern = 'ModelMavenPreviewLayout', callback = relayout})
  api.nvim_create_autocmd('BufWipeout', {buffer = buf, once = true, callback = function() pcall(api.nvim_del_augroup_by_id, resize_group) end})
  local function remember_view(force)
    if not alive() or not actions.view_changed then return end
    local view = api.nvim_win_call(win, vim.fn.winsaveview)
    actions.view_changed({selected = selected, topline = math.min(4096, math.max(1, view.topline)),
      leftcol = math.min(4096, math.max(0, view.leftcol or 0))}, force == true)
  end
  api.nvim_create_autocmd('CursorMoved', {buffer = buf, callback = function()
    remember_view(false)
  end})
  for key_name, direction in pairs({['<PageUp>'] = -1, ['<PageDown>'] = 1}) do
    vim.keymap.set('n', key_name, function()
      if not alive() then return end
      if dragging or picked then return render(tr('Finish or cancel the gesture first.', '请先完成或取消当前操作。', '請先完成或取消目前操作。')) end
      local view = api.nvim_win_call(win, vim.fn.winsaveview)
      local step = math.max(1, api.nvim_win_get_height(win) - 2)
      local last = api.nvim_buf_line_count(buf)
      view.topline = math.max(1, math.min(last, view.topline + direction * step))
      view.lnum = math.max(1, math.min(last, view.lnum + direction * step))
      api.nvim_win_call(win, function() vim.fn.winrestview(view) end)
      remember_view(false)
    end, {buffer = buf, nowait = true})
  end
  api.nvim_create_autocmd('WinLeave', {buffer = buf, callback = function()
    local cancelled = cancel_gesture() or picked ~= nil
    picked = nil
    remember_view(false)
    if cancelled and alive() then render(tr('Gesture cancelled on window change.', '窗口切换，操作已取消。', '視窗切換，操作已取消。')) end
  end})
  vim.keymap.set('n', 'q', function() if alive() then remember_view(true); api.nvim_win_close(win, true) end end, {buffer = buf})
  local function find_nodes()
    if dragging then return render(tr('Finish or cancel the gesture first.', '请先完成或取消当前操作。', '請先完成或取消目前操作。')) end
    local snapshot = document
    vim.ui.input({prompt = tr('Find node, component, factory or port: ', '查找节点、组件、工厂或端口：', '尋找節點、元件、工廠或連接埠：')}, function(raw)
      if not raw or not alive() or document ~= snapshot then return end
      local query = vim.trim(raw):lower()
      if query == '' then return end
      local definitions, results, labels = {}, {}, {}
      for _, definition in ipairs(document.definitions or {}) do definitions[definition.id] = definition end
      for _, node in ipairs(document.nodes) do
        local definition = node.definition and definitions[node.definition] or nil
        local text = table.concat({node.id, node.kind, node.definition or '', definition and definition.factory or '',
          table.concat(node.inputs or {}, ' '), table.concat(node.outputs or {}, ' ')}, ' ')
        if text:lower():find(query, 1, true) then
          results[#results + 1] = node.id
          labels[#labels + 1] = node.id .. ' · ' .. node.kind .. (definition and (' · ' .. definition.factory) or '')
        end
      end
      if #results == 0 then return render(tr('No matching nodes.', '没有匹配的节点。', '沒有符合的節點。')) end
      vim.ui.select(labels, {prompt = tr('Choose matching node', '选择匹配节点', '選擇符合節點')}, function(_, index)
        local id = index and results[index]
        if not id or not alive() or document ~= snapshot or not placed[id] then return end
        selected = id; render()
        local point = placed[id]
        api.nvim_win_set_cursor(win, {first_row + point.y * 2, point.x * cell + 3})
        api.nvim_win_call(win, function() vim.cmd('normal! zz') end)
      end)
    end)
  end
  local function command_catalog()
  local commands = {
    {'a', 'add', tr('Add node', '添加节点', '新增節點')},
    {'o', 'add_operator', tr('Add an operator', '添加算子', '新增運算子')},
    {'H', 'structure', tr('Choose basic structure', '选择基础结构', '選擇基礎結構')},
    {'G', 'groups', tr('Manage display groups', '管理显示分组', '管理顯示分組')},
    {'v', 'validate', tr('Plan and locate issues', '规划并定位问题', '規劃並定位問題')},
    {'s', 'save', tr('Save graph', '保存图', '儲存圖')},
    {'S', 'save_and_apply_data', tr('Save graph, then review project data', '保存图并继续审核项目数据', '儲存圖並繼續審核專案資料')},
    {'r', 'reconcile', tr('Compare graph', '比较图', '比較圖')},
    {'g', 'generate', tr('Generate source', '生成源码', '產生原始碼')},
    {'O', 'source', tr('Open source editor', '打开源码编辑器', '開啟原始碼編輯器')},
    {'x', 'source_to_node', tr('Locate source line in graph', '将源码行定位到图节点', '將原始碼列定位至圖節點')},
    {'b', 'apply_data', tr('Review project data', '审核项目数据', '審核專案資料')},
    {'B', 'bundle', tr('Package graph project', '打包图项目', '打包圖專案')},
    {'L', 'link_supervision', tr('Link supervision bundle', '关联监督声明', '連結監督宣告')},
    {'U', 'remove_supervision', tr('Remove supervision reference', '移除监督引用', '移除監督參照')},
    {'V', 'import_supervision', tr('Review and import linked supervision', '审核并导入关联监督', '檢視並匯入連結的監督')},
    {'X', 'export_supervision', tr('Export current supervision', '导出当前监督', '匯出目前監督')},
    {'p', 'forward', tr('Run forward', '运行前向', '執行前向')},
    {'w', 'observed_links', tr('Browse observed tensor links', '查看已观察的张量连接', '檢視已觀察的張量連線')},
    {'T', 'training', tr('Training settings', '训练设置', '訓練設定')},
    {'J', 'training_compare', tr('Compare graph and session training', '比较图与会话训练设置', '比較圖與工作階段訓練設定')},
    {'K', 'training_import', tr('Load session training into graph draft', '将会话训练设置载入图草稿', '將工作階段訓練設定載入圖草稿')},
    {'N', 'component_storage', tr('Component weight naming', '组件权重命名', '元件權重命名')},
    {'t', 'trial', tr('Prepare trial', '准备试跑', '準備試跑')},
    {'Y', 'train', tr('Prepare training', '准备训练', '準備訓練')},
    {'R', 'runs', tr('Browse project runs and results', '查看项目运行和结果', '查看專案執行和結果')},
    {'Q', 'pending_runs', tr('Inspect pending training request', '核查待确认训练请求', '核查待確認訓練要求')},
    {'i', 'capture_input', tr('Capture input', '捕获输入', '擷取輸入')},
    {'I', 'replay_input', tr('Replay input', '重放输入', '重播輸入')},
    {'D', 'discard_input', tr('Discard captured input', '丢弃输入快照', '丟棄輸入快照')},
    {'d', 'locate_failure', tr('Locate failure', '定位失败', '定位失敗')},
    {'<F2>', 'language', tr('Language', '语言', '語言')},
  }
  if actions.delivery_review then
    table.insert(commands, 8, {'C', 'delivery_review', tr('Review graph, source and project together', '联合审核图、源码和项目', '聯合檢視圖、原始碼和專案')})
  end
  if actions.delivery_resume then
    table.insert(commands, 9, {'Z', 'delivery_resume', tr('Resume recorded delivery', '恢复记录的交付', '恢復記錄的交付')})
  end
  return commands
  end
  local commands = command_catalog()
  local browse_connections
  local function command_menu()
    commands = command_catalog()
    local labels = {}; for _, item in ipairs(commands) do labels[#labels + 1] = '[' .. item[1] .. '] ' .. item[3] end
    labels[#labels + 1] = tr('[/] Find node', '[/] 查找节点', '[/] 尋找節點')
    labels[#labels + 1] = tr('[e] Node connections', '[e] 当前节点连接', '[e] 目前節點連線')
    labels[#labels + 1] = tr('[E] Search all connections', '[E] 搜索所有连接', '[E] 搜尋所有連線')
    vim.ui.select(labels, {prompt = tr('Graph actions', '图操作', '圖操作')}, function(_, choice)
      if choice and alive() then
        if choice == #commands + 1 then find_nodes()
        elseif choice == #commands + 2 then browse_connections(false)
        elseif choice == #commands + 3 then browse_connections(true)
        elseif choice <= #commands then actions.command(commands[choice][2], selected) end
      end
    end)
  end
  local function run_next_step(override)
    if not alive() or dragging then return end
    local _, command = next_step()
    command = override or command
    if command == 'connect_missing' then
      local snapshot, connected, destinations, labels = document, {}, {}, {}
      for _, edge in ipairs(document.edges or {}) do connected[edge.to.node .. '\0' .. edge.to.port] = true end
      for _, node in ipairs(document.nodes) do
        for _, port in ipairs(node.inputs or {}) do
          if not connected[node.id .. '\0' .. port] then
            destinations[#destinations + 1] = {node = node.id, port = port}
            labels[#labels + 1] = node.id .. '.' .. port
          end
        end
      end
      return vim.ui.select(labels, {prompt = tr('Choose an unconnected input', '选择未连接的输入端口', '選擇未連接的輸入連接埠')}, function(_, index)
        local target = index and destinations[index]
        if not target or not alive() or document ~= snapshot then return end
        local sources, source_labels = {}, {}
        for _, node in ipairs(document.nodes) do
          if node.id ~= target.node then
            for _, port in ipairs(node.outputs or {}) do
              sources[#sources + 1] = {node = node.id, port = port}
              source_labels[#source_labels + 1] = node.id .. '.' .. port
            end
          end
        end
        if #sources == 0 then return render(tr('Add a node with an output first.', '请先添加带输出的节点。', '請先新增帶輸出的節點。')) end
        vim.ui.select(source_labels, {prompt = tr('Choose a source for ', '选择来源：', '選擇來源：') .. target.node .. '.' .. target.port}, function(_, source_index)
          local source = source_index and sources[source_index]
          if source and alive() and document == snapshot then
            selected = target.node
            actions.connect_ports(source.node, source.port, target.node, target.port)
          end
        end)
      end)
    end
    if command ~= 'choose_data_node' and command ~= 'choose_objective_node' then
      if command then actions.command(command, selected) else command_menu() end
      return
    end
    local snapshot, ids, labels = document, {}, {}
    for _, node in ipairs(document.nodes) do
      if node.kind == 'output' or (command == 'choose_data_node' and node.kind == 'input') then
        ids[#ids + 1] = node.id
        labels[#labels + 1] = node.id .. ' · ' .. (node.kind == 'input'
          and tr('input', '输入', '輸入') or tr('output', '输出', '輸出'))
      end
    end
    local prompt = command == 'choose_data_node'
      and tr('Choose a node to connect data', '选择要连接数据的节点', '選擇要連接資料的節點')
      or tr('Choose an output to configure its loss', '选择要配置损失的输出', '選擇要設定損失的輸出')
    local function open(id)
      if id and alive() and document == snapshot and nodes[id] then
        selected = id
        render()
        actions.select(id, command == 'choose_data_node' and 'data' or 'objective')
      end
    end
    if #ids == 0 then return render(tr('Add the corresponding input or output node first.', '请先添加对应的输入或输出节点。', '請先新增對應的輸入或輸出節點。')) end
    if selected and vim.tbl_contains(ids, selected) then return open(selected) end
    if #ids == 1 then return open(ids[1]) end
    vim.ui.select(labels, {prompt = prompt}, function(_, choice)
      open(choice and ids[choice])
    end)
  end
  vim.keymap.set('n', 'm', command_menu, {buffer = buf})
  vim.keymap.set('n', 'n', function() run_next_step() end, {buffer = buf, nowait = true})
  vim.keymap.set('n', 'F', function()
    if not alive() then return end
    code_layout = not code_layout
    cancel_gesture()
    local next_width = code_layout and math.max(1, math.floor((vim.o.columns - 5) / 2)) or math.max(1, math.min(104, vim.o.columns - 4))
    local next_height = math.max(1, math.min(36, vim.o.lines - 5))
    pcall(api.nvim_win_set_config, win, {relative = 'editor', row = 1,
      col = code_layout and (source_on_right and 1 or math.max(0, vim.o.columns - next_width - 2)) or math.max(0, math.floor((vim.o.columns - next_width) / 2)),
      width = next_width, height = next_height})
    if actions.layout then actions.layout(code_layout, source_on_right) end
    render()
  end, {buffer = buf})
  vim.keymap.set('n', '/', find_nodes, {buffer = buf, nowait = true})
  for _, item in ipairs(commands) do
    local shortcut, command = item[1], item[2]
    vim.keymap.set('n', shortcut, function() if alive() then actions.command(command, selected) end end, {buffer = buf, nowait = true})
  end
  local function history(redo_action)
    if dragging then return render(tr('Finish or cancel the gesture first.', '请先完成或取消当前操作。', '請先完成或取消目前操作。')) end
    local next_document, next_positions, reason = actions.history(redo_action)
    if not next_document then return render(reason == 'stale' and tr('Graph changed; reopen this view.', '图已变化，请重新打开此视图。', '圖已變更，請重新開啟此檢視。')
      or tr('Nothing to restore.', '没有可恢复的修改。', '沒有可還原的修改。')) end
    update_layout(next_document, next_positions)
    render(redo_action and tr('Change redone.', '已重做。', '已重做。') or tr('Change undone.', '已撤销。', '已復原。'))
  end
  vim.keymap.set('n', 'u', function() history(false) end, {buffer = buf})
  vim.keymap.set('n', '<C-r>', function() history(true) end, {buffer = buf})
  vim.keymap.set('n', 'A', arrange, {buffer = buf})
  vim.keymap.set('n', 'f', function()
    if not selected then return render(tr('Select a node before focusing.', '聚焦前请先选择节点。', '聚焦前請先選取節點。')) end
    focused = not focused
    render()
  end, {buffer = buf})
  browse_connections = function(search_all)
    local snapshot = document
    local selected_node = selected
    local function show(query)
      if not alive() or document ~= snapshot then return end
      query = (query or ''):lower()
      local labels, indices = {}, {}
      for index, edge in ipairs(document.edges or {}) do
        local label = edge.from.node .. '.' .. edge.from.port .. ' → ' .. edge.to.node .. '.' .. edge.to.port
        if (search_all or not selected_node or edge.from.node == selected_node or edge.to.node == selected_node)
            and (query == '' or label:lower():find(query, 1, true)) then
          labels[#labels + 1] = label
          indices[#indices + 1] = index
        end
      end
      if #labels == 0 then return render(tr('No matching connections.', '没有匹配的连接。', '沒有符合的連線。')) end
      vim.ui.select(labels, {prompt = tr('Choose connection · matches: ', '选择连接 · 匹配数：', '選擇連線 · 符合數：') .. #labels}, function(_, choice)
        if choice and alive() and document == snapshot then actions.edge(indices[choice]) end
      end)
    end
    if search_all then
      vim.ui.input({prompt = tr('Find connection by node or port (blank: all): ', '按节点或端口搜索连接（留空显示全部）：', '依節點或連接埠搜尋連線（留空顯示全部）：')}, function(query)
        if query ~= nil then show(vim.trim(query)) end
      end)
    else show('') end
  end
  vim.keymap.set('n', 'e', function() browse_connections(false) end, {buffer = buf})
  vim.keymap.set('n', 'E', function() browse_connections(true) end, {buffer = buf})
  vim.keymap.set('n', 'c', function()
    if not selected or #(nodes[selected].outputs or {}) == 0 then
      return render(tr('Select a node with an output first.', '请先选中有输出的节点。', '請先選取有輸出的節點。'))
    end
    local targets = {}
    for _, node in ipairs(document.nodes) do if node.id ~= selected and #(node.inputs or {}) > 0 then targets[#targets + 1] = node.id end end
    if #targets == 0 then return render(tr('No input nodes available.', '没有可用的输入节点。', '沒有可用的輸入節點。')) end
    local source, snapshot = selected, document
    vim.ui.select(targets, {prompt = tr('Choose destination node', '选择目标节点', '選擇目標節點')}, function(target)
      if target and alive() and document == snapshot then actions.connect(source, target) end
    end)
  end, {buffer = buf})
  local function select_at(id)
    if not id or not alive() or not placed[id] then return end
    selected = id
    render()
    local point = placed[id]
    api.nvim_win_set_cursor(win, {first_row + point.y * 2, point.x * cell + 3})
    remember_view(false)
  end
  local function ordered_nodes()
    local result = {}
    for id, point in pairs(placed) do result[#result + 1] = {id = id, x = point.x, y = point.y} end
    table.sort(result, function(a, b)
      if a.y ~= b.y then return a.y < b.y end
      if a.x ~= b.x then return a.x < b.x end
      return a.id < b.id
    end)
    return result
  end
  local function cycle_selection(direction)
    if dragging then return render(tr('Finish or cancel the gesture first.', '请先完成或取消当前操作。', '請先完成或取消目前操作。')) end
    local list = ordered_nodes()
    if #list == 0 then return render(tr('No nodes to select.', '没有可选择的节点。', '沒有可選取的節點。')) end
    local current
    for index, item in ipairs(list) do if item.id == selected then current = index; break end end
    local index = current and ((current - 1 + direction) % #list + 1) or (direction > 0 and 1 or #list)
    select_at(list[index].id)
  end
  local function directional_selection(dx, dy)
    if dragging then return render(tr('Finish or cancel the gesture first.', '请先完成或取消当前操作。', '請先完成或取消目前操作。')) end
    if not selected or not placed[selected] then return cycle_selection(1) end
    local origin, best, best_score, best_primary = placed[selected], nil, nil, nil
    for id, point in pairs(placed) do
      local x, y = point.x - origin.x, point.y - origin.y
      local primary = x * dx + y * dy
      if primary > 0 then
        local secondary = math.abs(x * dy - y * dx)
        local score = primary + secondary
        if not best or score < best_score or (score == best_score and (primary < best_primary
            or (primary == best_primary and id < best))) then
          best, best_score, best_primary = id, score, primary
        end
      end
    end
    if best then select_at(best) else render(tr('No node in that direction; Tab cycles all nodes.', '该方向没有节点；Tab 可循环选择。', '該方向沒有節點；Tab 可循環選取。')) end
  end
  vim.keymap.set('n', '<Tab>', function() cycle_selection(1) end, {buffer = buf, nowait = true})
  vim.keymap.set('n', '<S-Tab>', function() cycle_selection(-1) end, {buffer = buf, nowait = true})
  for key_name, delta in pairs({['<Left>']={-1,0}, ['<Right>']={1,0}, ['<Up>']={0,-1}, ['<Down>']={0,1}}) do
    local dx, dy = delta[1], delta[2]
    vim.keymap.set('n', key_name, function() directional_selection(dx, dy) end, {buffer = buf, nowait = true})
  end
  for key_name, delta in pairs({h={-1,0}, j={0,1}, k={0,-1}, l={1,0}}) do
    vim.keymap.set('n', key_name, function()
      if picked then directional_selection(delta[1], delta[2])
      elseif selected and placed[selected] then move(selected, placed[selected].x + delta[1], placed[selected].y + delta[2], true) end
    end, {buffer = buf})
  end
  vim.keymap.set('n', 'M', function()
    if dragging then return render(tr('Finish or cancel the gesture first.', '请先完成或取消当前操作。', '請先完成或取消目前操作。')) end
    if picked then picked = nil; return render(tr('Reorder cancelled.', '已取消重排。', '已取消重排。')) end
    if not selected or not nodes[selected] or nodes[selected].kind ~= 'call' then
      return render(tr('Select an operator to reorder.', '请先选择要重排的算子。', '請先選擇要重排的運算子。'))
    end
    picked = {id = selected, snapshot = document}
    render()
  end, {buffer = buf, nowait = true})
  vim.keymap.set('n', '<LeftMouse>', function()
    local mouse = vim.fn.getmousepos(); if mouse.winid ~= win then return end
    if picked then
      local x, y = cell_at(mouse.line, mouse.column - 1)
      local id = x and occupied[key(x, y)]
      if id then select_at(id) end
      return
    end
    if mouse.line == 2 then
      for _, hit in ipairs(toolbar_hits) do
        if mouse.column >= hit.first and mouse.column <= hit.last then
          if hit.command == 'menu' then command_menu()
          elseif hit.command == 'choose_data_node' then run_next_step(hit.command)
          else actions.command(hit.command, selected) end
          return
        end
      end
      return
    end
    if mouse.line == 3 then
      run_next_step()
      return
    end
    if (mouse.line == 4 or mouse.line == 5) and selected then actions.select(selected); return end
    local column = mouse.column - 1
    local hits = wire_hits[mouse.line .. ':' .. column]
    if hits and #hits > 0 then
      local snapshot = document
      local labels = {}
      for _, index in ipairs(hits) do
        local edge = document.edges[index]
        labels[#labels + 1] = edge.from.node .. '.' .. edge.from.port .. ' → ' .. edge.to.node .. '.' .. edge.to.port
      end
      vim.ui.select(labels, {prompt = tr('Choose connection', '选择连接', '選擇連線')}, function(_, choice)
        if choice and alive() and document == snapshot then actions.edge(hits[choice]) end
      end)
      return
    end
    local x, y = cell_at(mouse.line, column); local id = x and occupied[key(x, y)]
    if id then
      local output_hit = (mouse.line - first_row) % 2 == 0 and column % cell == 22 and #(nodes[id].outputs or {}) > 0
      selected, dragging = id, {id = id, moved = false, mode = output_hit and 'wire' or 'node'}; render()
      remember_view(false)
    end
  end, {buffer = buf})
  vim.keymap.set('n', '<LeftDrag>', function()
    if not dragging or not alive() then return end
    local mouse = vim.fn.getmousepos()
    dragging.moved = true
    if mouse.winid ~= win then stop_scroll(); dragging.outside = true; drag_preview(nil, nil, true); return end
    dragging.outside = false
    local view = api.nvim_win_call(win, vim.fn.winsaveview)
    dragging.viewport_row = math.max(1, mouse.line - view.topline + 1)
    dragging.column = math.max(0, mouse.column - 1)
    drag_preview(mouse.line, dragging.column, false)
    local window_height = api.nvim_win_get_height(win)
    local window_width = api.nvim_win_get_width(win)
    local screen_col = type(mouse.wincol) == 'number' and mouse.wincol or 1
    scroll_drag(dragging.viewport_row <= 1 and -1 or (dragging.viewport_row >= window_height and 1 or 0),
      screen_col <= 1 and -1 or (screen_col >= window_width and 1 or 0))
  end, {buffer = buf})
  vim.keymap.set('n', '<LeftRelease>', function()
    if not dragging then return end
    local item, mouse = dragging, vim.fn.getmousepos(); dragging = nil; stop_scroll(); api.nvim_buf_clear_namespace(buf, preview_namespace, 0, -1)
    if mouse.winid ~= win or item.outside then return render(tr('Move cancelled.', '移动已取消。', '移動已取消。')) end
    if item.moved and not item.valid_grid then return render(tr('Drop cancelled outside the available grid.', '目标不在可用网格中，操作已取消。', '目標不在可用網格中，操作已取消。')) end
    local view = api.nvim_win_call(win, vim.fn.winsaveview)
    local row = item.viewport_row and view.topline + item.viewport_row - 1 or mouse.line
    local column = item.column or mouse.column - 1
    if item.moved and item.mode == 'wire' then
      local x, y = cell_at(row, column); local id = x and occupied[key(x, y)]
      if id and id ~= item.id and (row - first_row) % 2 == 0 and column % cell == 1 and #(nodes[id].inputs or {}) > 0 then
        actions.connect(item.id, id)
      else render(tr('Connection cancelled; drop o on an input i.', '接线已取消；请将 o 拖到输入 i。', '接線已取消；請將 o 拖到輸入 i。')) end
    elseif item.moved then local x, y = cell_at(row, column); if x then move(item.id, x, y, false, actions.reorder_after == true and column % cell >= math.floor(cell / 2)) end
    else actions.select(item.id) end
  end, {buffer = buf})
  vim.keymap.set('n', '<Esc>', function()
    if dragging then cancel_gesture(); render(tr('Gesture cancelled.', '操作已取消。', '操作已取消。')) end
    if picked then picked = nil; render(tr('Reorder cancelled.', '已取消重排。', '已取消重排。')) end
  end, {buffer = buf})
  vim.keymap.set('n', '<CR>', function()
    local row, col = unpack(api.nvim_win_get_cursor(win)); local x, y = cell_at(row, col)
    local id = x and occupied[key(x, y)] or selected
    if picked then
      if picked.snapshot ~= document then picked = nil; return render(tr('Graph changed; pick the node again.', '图已变化，请重新拾取节点。', '圖已變更，請重新拾取節點。')) end
      if not id or id == picked.id or not placed[id] then return render(tr('Choose a different target node.', '请选择另一个目标节点。', '請選擇另一個目標節點。')) end
      local source = picked.id; picked = nil
      return move(source, placed[id].x, placed[id].y)
    end
    if id then actions.select(id) end
  end, {buffer = buf})
  vim.keymap.set('n', '<Space>', function()
    local row, col = unpack(api.nvim_win_get_cursor(win)); local x, y = cell_at(row, col)
    local id = x and occupied[key(x, y)]
    if id then selected = id; render(tr('Node selected for keyboard movement.', '已选中节点，可用键盘移动。', '已選取節點，可用鍵盤移動。')) end
  end, {buffer = buf})
  if type(actions.view_state) == 'table' and type(actions.view_state.selected) == 'string'
      and placed[actions.view_state.selected] then selected = actions.view_state.selected end
  render()
  if type(actions.view_state) == 'table' then
    local restored = actions.view_state
    local view = api.nvim_win_call(win, vim.fn.winsaveview)
    if selected and placed[selected] then
      local point = placed[selected]
      view.lnum, view.col = first_row + point.y * 2, point.x * cell + 3
    end
    if type(restored.topline) == 'number' then view.topline = math.min(api.nvim_buf_line_count(buf), math.max(1, restored.topline)) end
    if type(restored.leftcol) == 'number' then view.leftcol = math.max(0, math.min(4096, restored.leftcol)) end
    api.nvim_win_call(win, function() vim.fn.winrestview(view) end)
  end
  return {alive = alive, show_source = function(source_right)
    if not alive() then return end
    code_layout = true
    source_on_right = source_right == true
    cancel_gesture()
    local next_width = math.max(1, math.floor((vim.o.columns - 5) / 2))
    local next_height = math.max(1, math.min(36, vim.o.lines - 5))
    pcall(api.nvim_win_set_config, win, {relative = 'editor', row = 1, col = source_on_right and 1 or math.max(0, vim.o.columns - next_width - 2),
      width = next_width, height = next_height})
    render()
  end, status = function(info, focus)
    if not alive() then return end
    if focus and nodes[focus] then selected = focus end
    render(type(info) == 'string' and info:gsub('[%c]', ' ') or nil)
    if focus and placed[focus] then
      local point = placed[focus]
      api.nvim_win_set_cursor(win, {first_row + point.y * 2, point.x * cell + 3})
      api.nvim_win_call(win, function() vim.cmd('normal! zz') end)
    end
  end, refresh = function(next_doc, next_positions)
    if not alive() then return end
    if dragging then cancel_gesture() end
    picked = nil
    update_layout(next_doc, next_positions)
    render()
  end}
end

return M
