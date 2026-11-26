-- A read-only view of the graph declaration, independent of runtime probing.
local M, api = {}, vim.api
local tr = require('model_maven.locale').text
local function supports_live_renderer()
  return vim.fn.has('unix') == 1 or vim.fn.has('win32') == 1
end
function M.layout(limit, boundary)
  local available = vim.o.columns
  if type(boundary) == 'function' then
    local ok, col = pcall(boundary)
    if ok and type(col) == 'number' and col >= 42 then available = col end
  end
  local width = math.max(1, math.min(limit, available - 4))
  return width, math.max(0, math.floor((available - width) / 2))
end

-- Reports and reviews remain readable when the floating preview appears,
-- moves or changes size. Compact previews live in a split and need no space
-- reserved here. Only our own auxiliary floats are repositioned.
function M.reflow_aux()
  local boundary
  for _, win in ipairs(api.nvim_tabpage_list_wins(0)) do
    if api.nvim_win_is_valid(win) and vim.b[api.nvim_win_get_buf(win)].model_maven_preview
        and api.nvim_win_get_config(win).relative ~= '' then
      local col = api.nvim_win_get_position(win)[2]
      boundary = boundary and math.min(boundary, col) or col
    end
  end
  local width, col = M.layout(110, function() return boundary end)
  local height = math.min(32, vim.o.lines - 5)
  if width < 20 or height < 4 then return end
  for _, win in ipairs(api.nvim_tabpage_list_wins(0)) do
    if api.nvim_win_is_valid(win) and vim.b[api.nvim_win_get_buf(win)].model_maven_aux_window
        and api.nvim_win_get_config(win).relative ~= '' then
      pcall(api.nvim_win_set_config, win, {relative='editor', row=1, col=col, width=width, height=height})
    end
  end
  api.nvim_exec_autocmds('User', {pattern='ModelMavenPreviewLayout', modeline=false})
end
local function text(value)
  return tostring(value or '?'):gsub('[%c]', ' ')
end

local function graph(state)
  local session = state.graph_canvas
  if session and session.project == state.opts.project and type(session.draft) == 'table' then
    local saved = session.base ~= 'absent' and vim.deep_equal(session.draft, session.saved_doc)
    return session.draft, saved and tr('Saved declaration · unverified', '已保存声明 · 未验证', '已儲存宣告 · 未驗證')
      or tr('Draft · unverified', '草稿 · 未验证', '草稿 · 未驗證')
  end
  local path = state.opts.project .. '/model-graph.json'
  local stat = vim.uv.fs_stat(path)
  if stat and stat.type == 'file' and stat.size <= 2 * 1024 * 1024 then
    local file = io.open(path, 'rb')
    if file then
      local raw = file:read('*a'); file:close()
      local ok, document = pcall(vim.json.decode, raw)
      if ok and type(document) == 'table' and type(document.nodes) == 'table'
          and type(document.edges) == 'table' then
        return document, tr('Saved declaration · unverified', '已保存声明 · 未验证', '已儲存宣告 · 未驗證')
      end
    end
    return nil
  end
  local observation = state.model_preview_observation
  local report = observation and observation.report
  local modules = report and type(report.data) == 'table' and report.data.modules
  if observation and observation.project == state.opts.project and type(modules) == 'table' then
    local nodes, seen = {}, {}
    for _, module in ipairs(modules) do
      if type(module) == 'table' and type(module.path) == 'string' and module.path ~= ''
          and #module.path <= 128 and not module.path:find('[%c]') and not seen[module.path] then
        nodes[#nodes+1] = {id=module.path, kind=type(module.type) == 'string' and module.type or '?', inputs={}, outputs={}}
        seen[module.path] = true
      end
    end
    if #nodes > 0 then
      local stale = observation.epoch ~= (state.change_epoch or 0) or observation.context_changed
      return {nodes=nodes, definitions={}, edges={}}, stale
        and tr('Inspection snapshot · stale · links unknown', '检查快照 · 已过期 · 连线未知', '檢查快照 · 已過期 · 連線未知')
        or tr('Inspection snapshot · links unknown', '检查快照 · 连线未知', '檢查快照 · 連線未知')
    end
  end
end

local function lines(state)
  local document, source = graph(state)
  local result = {tr('◈ Model preview  [8] hide', '◈ 模型预览  [8] 隐藏', '◈ 模型預覽  [8] 隱藏')}
  if not document then
    result[#result+1] = tr('No graph declaration yet.', '尚无模型图声明。', '尚無模型圖宣告。')
    result[#result+1] = tr('[n] Design a model', '[n] 开始设计模型', '[n] 開始設計模型')
    return result
  end
  result[#result+1] = source
  if not supports_live_renderer() then
    result[#result+1] = tr('Interactive 3D unavailable here',
      '此平台暂不支持交互 3D', '此平台暫不支援互動 3D')
  elseif state.model_preview_live_unavailable then
    result[#result+1] = tr('Interactive 3D unavailable',
      '交互 3D 不可用', '互動 3D 無法使用')
  elseif vim.o.columns < 70 or vim.o.lines < 15 then
    result[#result+1] = tr('Compact · enlarge for 3D', '紧凑预览 · 放大启用 3D', '緊湊預覽 · 放大啟用 3D')
  end
  local view = state.model_preview
  if view and view.art then
    for _, line in ipairs(view.art) do result[#result+1] = line end
    if view.art_visible and view.art_total and view.art_visible < view.art_total then
      result[#result+1] = tr('Partial 3D · Tab next · list below', '3D 局部 · Tab 切换 · 下方完整列表', '3D 局部 · Tab 切換 · 下方完整清單')
    end
  end
  local nodes, incoming, outgoing, degree, definitions = {}, {}, {}, {}, {}
  for _, d in ipairs(document.definitions or {}) do if type(d) == 'table' then definitions[d.id] = d end end
  for _, n in ipairs(document.nodes) do
    if type(n) == 'table' and type(n.id) == 'string' and not nodes[n.id] then
      nodes[n.id] = n; incoming[n.id] = {}; outgoing[n.id] = {}; degree[n.id] = 0
    end
  end
  for _, e in ipairs(document.edges) do
    if type(e) == 'table' and type(e.from) == 'table' and type(e.to) == 'table'
        and nodes[e.from.node] and nodes[e.to.node] then
      incoming[e.to.node][#incoming[e.to.node]+1] = e
      outgoing[e.from.node][#outgoing[e.from.node]+1] = e.to.node
      degree[e.to.node] = degree[e.to.node] + 1
    end
  end
  local ready, ordered, seen = {}, {}, {}
  for id, count in pairs(degree) do if count == 0 then ready[#ready+1] = id end end
  table.sort(ready)
  while #ready > 0 do
    local id = table.remove(ready, 1)
    if not seen[id] then
      seen[id] = true; ordered[#ordered+1] = id
      for _, target in ipairs(outgoing[id]) do
        degree[target] = degree[target] - 1
        if degree[target] == 0 then ready[#ready+1] = target; table.sort(ready) end
      end
    end
  end
  local remaining = {}
  for id in pairs(nodes) do if not seen[id] then remaining[#remaining+1] = id end end
  table.sort(remaining); vim.list_extend(ordered, remaining)
  for index, id in ipairs(ordered) do
    if index > 256 then result[#result+1] = tr('… more nodes', '… 其余节点', '… 其餘節點'); break end
    local node = nodes[id]
    local definition = node.definition and definitions[node.definition]
    local kind = definition and type(definition.factory) == 'string' and definition.factory:match('([^:%.]+)$') or node.kind
    if #incoming[id] > 0 then
      for _, edge in ipairs(incoming[id]) do
        result[#result+1] = '  ↳ ' .. text(edge.from.node) .. '.' .. text(edge.from.port)
          .. ' → ' .. text(edge.to.port)
      end
    end
    result[#result+1] = '  ◆ ' .. text(id) .. '  [' .. text(kind) .. ']'
  end
  if #ordered == 0 then result[#result+1] = tr('Empty graph · [n] add nodes', '空图 · [n] 添加节点', '空圖 · [n] 新增節點') end
  result[#result+1] = tr('↵ Open graph editor', '↵ 打开图编辑器', '↵ 開啟圖編輯器')
  return result
end

function M.refresh(state)
  local view = state.model_preview
  if not view or state.closed or not api.nvim_buf_is_valid(view.buf) then return end
  local document, source = graph(state)
  if view.live then
    if document then view.update(document, source) end
    return
  end
  local renderer = vim.env.MODEL_MAVEN_TUI_RENDERER
  if not renderer or renderer == '' then renderer = vim.fn.exepath('m-model-tui') end
  if document and #document.nodes > 0 and renderer and renderer ~= '' and supports_live_renderer()
      and vim.o.columns >= 70 and vim.o.lines >= 15 and not view.upgrade_pending and not state.model_preview_live_unavailable then
    view.upgrade_pending = true
    vim.schedule(function()
      if not state.closed and state.model_preview == view then M.close(state); M.open(state) end
    end)
    return
  end
  if document and type(renderer) == 'string' and renderer ~= '' and view.win and api.nvim_win_is_valid(view.win) then
    local width = math.max(20, math.min(100, api.nvim_win_get_width(view.win)))
    local ok, payload = pcall(vim.json.encode, {document=document, width=width, height=18, selected=view.selected or 0})
    if ok and #payload <= 2500000 then
      local key = vim.fn.sha256(payload)
      if view.render_key ~= key then
        if view.job then pcall(view.job.kill, view.job, 15); view.job = nil end
        view.render_key, view.art, view.art_visible, view.art_total = key, nil, nil, nil
        local started, job = pcall(vim.system, {renderer, '--render-model-preview'}, {stdin=payload, text=true}, function(result)
          vim.schedule(function()
            if state.closed or state.model_preview ~= view or view.render_key ~= key or not api.nvim_buf_is_valid(view.buf) then return end
            view.job = nil
            if result.code == 0 and type(result.stdout) == 'string' and #result.stdout <= 65536 then
              local parsed, data = pcall(vim.json.decode, result.stdout)
              if parsed and type(data) == 'table' and type(data.art) == 'string' then
                view.art = vim.split(data.art:gsub('\n$', ''), '\n', {plain=true})
                for index, line in ipairs(view.art) do view.art[index] = line:gsub('%s+$', '') end
                view.art_visible, view.art_total = data.visible, data.nodes
              end
            end
            M.refresh(state)
          end)
        end)
        if started then view.job = job end
      end
    end
  end
  vim.bo[view.buf].modifiable = true
  api.nvim_buf_set_lines(view.buf, 0, -1, false, lines(state))
  vim.bo[view.buf].modifiable = false
end
function M.close(state)
  local view = state.model_preview
  state.model_preview = nil
  if not view then return end
  if view.live then view.close(); M.reflow_aux(); return end
  if view.job then pcall(view.job.kill, view.job, 15); view.job = nil end
  if view.win and api.nvim_win_is_valid(view.win) and api.nvim_win_get_buf(view.win) == view.buf then
    pcall(api.nvim_win_close, view.win, true)
  end
  if api.nvim_buf_is_valid(view.buf) then pcall(api.nvim_buf_delete, view.buf, {force=true}) end
end
function M.open(state)
  if state.closed or state.model_preview or not api.nvim_win_is_valid(state.panel_win) then return end
  local document, source = graph(state)
  local renderer = vim.env.MODEL_MAVEN_TUI_RENDERER
  if not renderer or renderer == '' then renderer = vim.fn.exepath('m-model-tui') end
  if document and #document.nodes > 0 and renderer and renderer ~= '' and supports_live_renderer()
      and vim.o.columns >= 70 and vim.o.lines >= 15 and not state.model_preview_live_unavailable then
    local live = require('model_maven.model_live_preview').open(state, document, source, renderer)
    if live then state.model_preview = live; M.reflow_aux(); return end
    state.model_preview_live_unavailable = true
  end
  local height = api.nvim_win_get_height(state.panel_win)
  if height < 7 then return end
  local previous = api.nvim_get_current_win()
  local buf = api.nvim_create_buf(false, true)
  vim.b[buf].model_maven_preview = true
  vim.bo[buf].buftype, vim.bo[buf].bufhidden, vim.bo[buf].swapfile = 'nofile', 'wipe', false
  local before = {}; for _, win in ipairs(api.nvim_tabpage_list_wins(state.tab)) do before[win] = true end
  local ok = pcall(api.nvim_win_call, state.panel_win, function()
    vim.cmd('noautocmd belowright ' .. math.min(height - 3, math.max(3, math.min(13, math.floor(height * 0.4)))) .. 'split')
  end)
  local win
  for _, candidate in ipairs(api.nvim_tabpage_list_wins(state.tab)) do
    if not before[candidate] then if win then win = nil; break else win = candidate end end
  end
  if not ok or not win or not api.nvim_win_is_valid(win) then pcall(api.nvim_buf_delete, buf, {force=true}); return end
  if not pcall(api.nvim_win_set_buf, win, buf) or not api.nvim_win_is_valid(win) then
    pcall(api.nvim_buf_delete, buf, {force=true}); return
  end
  state.model_preview = {buf=buf, win=win}
  vim.wo[win].wrap = false
  vim.wo[win].number, vim.wo[win].relativenumber, vim.wo[win].signcolumn = false, false, 'no'
  vim.keymap.set('n', '8', function() state.model_preview_hidden = true; M.close(state) end, {buffer=buf, silent=true})
  vim.keymap.set('n', 'q', function() state.model_preview_hidden = true; M.close(state) end, {buffer=buf, silent=true})
  local function select_node(delta)
    local document = graph(state)
    local count = document and type(document.nodes) == 'table' and #document.nodes or 0
    if count > 0 and state.model_preview then
      state.model_preview.selected = ((state.model_preview.selected or 0) + delta) % count
      M.refresh(state)
    end
  end
  vim.keymap.set('n', '<Tab>', function() select_node(1) end, {buffer=buf, silent=true})
  vim.keymap.set('n', '<S-Tab>', function() select_node(-1) end, {buffer=buf, silent=true})
  vim.keymap.set('n', '<CR>', function() if state.actions and state.actions.n then state.actions.n() end end, {buffer=buf, silent=true})
  vim.keymap.set('n', '<LeftMouse>', function()
    local mouse = vim.fn.getmousepos()
    if mouse.winid ~= win then return '<LeftMouse>' end
    if mouse.line == 1 then
      vim.schedule(function() state.model_preview_hidden = true; M.close(state) end)
      return '<Ignore>'
    end
    if mouse.line == api.nvim_buf_line_count(buf) then
      vim.schedule(function() if state.actions and state.actions.n then state.actions.n() end end)
      return '<Ignore>'
    end
    return '<LeftMouse>'
  end, {buffer=buf, expr=true, silent=true})
  local group = api.nvim_create_augroup('ModelMavenPreview' .. buf, {clear=true})
  api.nvim_create_autocmd('VimResized', {group=group, callback=function()
    if state.model_preview and state.model_preview.buf == buf then M.refresh(state) end
  end})
  api.nvim_create_autocmd('BufWipeout', {group=group, buffer=buf, once=true, callback=function()
    if state.model_preview and state.model_preview.buf == buf then
      local job = state.model_preview.job
      state.model_preview = nil; state.model_preview_hidden = true
      if job then pcall(job.kill, job, 15) end
    end
    pcall(api.nvim_del_augroup_by_id, group)
  end})
  M.refresh(state)
  if api.nvim_win_is_valid(previous) then pcall(api.nvim_set_current_win, previous) end
end
return M
