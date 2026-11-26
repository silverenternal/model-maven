-- Movable floating window hosting the original interactive Rust 3D view.
local M, api, uv = {}, vim.api, vim.uv
local tr = require('model_maven.locale').text
local function clamp(n, lo, hi) return math.max(lo, math.min(hi, n)) end

function M.open(state, document, source, renderer)
  if vim.o.columns < 70 or vim.o.lines < 15 then return nil end
  local dir = vim.fn.tempname()
  if not uv.fs_mkdir(dir, 448) then return nil end
  local path = dir .. '/graph.json'
  local view = {live=true, dir=dir, path=path, document_key=nil}
  -- Give the model room to be inspected on large terminals as well as small
  -- ones; an absolute size ceiling wasted most of an ultrawide display.
  local height = clamp(math.floor(vim.o.lines * .72), 12, vim.o.lines - 4)
  -- Leave enough horizontal room for an operable graph editor on ordinary
  -- terminals. Users can still enlarge or move the window deliberately.
  local width = clamp(math.floor(vim.o.columns * .53), 24, vim.o.columns - 46)
  view.row, view.col, view.width, view.height = 1, math.max(1, vim.o.columns - width - 2), width, height
  local function write(doc, label)
    local ok, raw = pcall(vim.json.encode, {document=doc, source=label, locale=require('model_maven.locale').current()})
    if not ok or #raw > 2500000 then return false end
    local key = vim.fn.sha256(raw)
    if key == view.document_key then return true end
    local tmp = path .. '.new'
    local fd = uv.fs_open(tmp, 'w', 384)
    if not fd then return false end
    local count = uv.fs_write(fd, raw, 0); uv.fs_close(fd)
    if count ~= #raw or not uv.fs_rename(tmp, path) then pcall(uv.fs_unlink, tmp); return false end
    view.document_key = key
    return true
  end
  if not write(document, source) then pcall(uv.fs_rmdir, dir); return nil end
  local header, body = api.nvim_create_buf(false, true), api.nvim_create_buf(false, true)
  vim.b[header].model_maven_preview = true
  vim.b[body].model_maven_preview = true
  vim.bo[header].bufhidden, vim.bo[body].bufhidden = 'wipe', 'wipe'
  local function header_text(label)
    local title = tr('⠿ 3D MODEL · drag to move', '⠿ 模型 3D · 拖动移动', '⠿ 模型 3D · 拖動移動')
    local suffix = tr(' [−] [+] [×]', ' [−] [+] [×]', ' [−] [+] [×]')
    local displayed_label = tostring(label or '')
    if view.width < 72 then
      title = '⠿ 3D'
      local short_labels = {
        [tr('Saved declaration · unverified', '已保存声明 · 未验证', '已儲存宣告 · 未驗證')] = tr('Saved · unverified', '已保存·未验证', '已儲存·未驗證'),
        [tr('Draft · unverified', '草稿 · 未验证', '草稿 · 未驗證')] = tr('Draft · unverified', '草稿·未验证', '草稿·未驗證'),
        [tr('Inspection snapshot · links unknown', '检查快照 · 连线未知', '檢查快照 · 連線未知')] = tr('Obs·links?', '观察·连线未知', '觀察·連線未知'),
        [tr('Inspection snapshot · stale · links unknown', '检查快照 · 已过期 · 连线未知', '檢查快照 · 已過期 · 連線未知')] = tr('Stale · links?', '观察·过期·未知', '觀察·過期·未知'),
      }
      displayed_label = short_labels[displayed_label] or displayed_label
    end
    local text = title .. ' · ' .. displayed_label
    while vim.fn.strdisplaywidth(text .. suffix) > view.width and vim.fn.strchars(text) > 0 do
      text = vim.fn.strcharpart(text, 0, vim.fn.strchars(text)-1)
    end
    vim.bo[header].modifiable = true
    api.nvim_buf_set_lines(header, 0, -1, false, {text .. suffix})
    vim.bo[header].modifiable = false
    view.source = label
  end
  header_text(source)
  local ok, hwin = pcall(api.nvim_open_win, header, false, {relative='editor', style='minimal', border='rounded',
    row=view.row, col=view.col, width=width, height=1, zindex=70})
  if not ok then pcall(api.nvim_buf_delete, header, {force=true}); pcall(api.nvim_buf_delete, body, {force=true}); pcall(uv.fs_unlink,path); pcall(uv.fs_rmdir,dir); return nil end
  local opened, bwin = pcall(api.nvim_open_win, body, false, {relative='editor', style='minimal', border='rounded',
    row=view.row+2, col=view.col, width=width, height=height-3, zindex=70})
  if not opened then pcall(api.nvim_win_close,hwin,true); pcall(api.nvim_buf_delete,body,{force=true}); pcall(uv.fs_unlink,path); pcall(uv.fs_rmdir,dir); return nil end
  view.header, view.buf, view.win, view.body_win = header, body, hwin, bwin
  local function position()
    if not api.nvim_win_is_valid(hwin) or not api.nvim_win_is_valid(bwin) then return end
    view.width = clamp(view.width, 24, math.max(24, vim.o.columns - 4))
    view.height = clamp(view.height, 9, math.max(9, vim.o.lines - 4))
    view.col = clamp(view.col, 0, math.max(0, vim.o.columns - view.width - 2))
    view.row = clamp(view.row, 0, math.max(0, vim.o.lines - view.height - 1))
    pcall(api.nvim_win_set_config, hwin, {relative='editor', row=view.row, col=view.col, width=view.width, height=1})
    pcall(api.nvim_win_set_config, bwin, {relative='editor', row=view.row+2, col=view.col, width=view.width, height=view.height-3})
    if api.nvim_buf_is_valid(header) then header_text(view.source) end
    require('model_maven.model_preview').reflow_aux()
  end
  view.reposition = position
  local function close()
    if view.closed then return end
    view.closed = true
    if view.job and view.job > 0 then pcall(vim.fn.jobstop, view.job) end
    if api.nvim_win_is_valid(bwin) then pcall(api.nvim_win_close, bwin, true) end
    if api.nvim_win_is_valid(hwin) then pcall(api.nvim_win_close, hwin, true) end
    if api.nvim_buf_is_valid(body) then pcall(api.nvim_buf_delete, body, {force=true}) end
    if api.nvim_buf_is_valid(header) then pcall(api.nvim_buf_delete, header, {force=true}) end
    pcall(uv.fs_unlink, path); pcall(uv.fs_unlink, path..'.new'); pcall(uv.fs_rmdir,dir)
  end
  view.close = close
  view.update = function(doc, label)
    local ok = write(doc, label)
    if ok and label ~= view.source and api.nvim_buf_is_valid(header) then header_text(label) end
    return ok
  end
  local drag
  local function mouse(key)
    local pos = vim.fn.getmousepos()
    if key == '<LeftMouse>' then
      if pos.winid ~= hwin then return key end
      local line = api.nvim_buf_get_lines(header, 0, 1, false)[1] or ''
      local label = vim.fn.strcharpart(line, math.max(0, vim.fn.strchars(line)-13))
      local suffix_start = vim.fn.strdisplaywidth(line) - vim.fn.strdisplaywidth(label)
      local x = pos.screencol - view.col - 2
      if x >= suffix_start and line:find('%[×%]') and x >= vim.fn.strdisplaywidth(line)-3 then
        vim.schedule(function() state.model_preview_hidden=true; require('model_maven.model_preview').close(state) end)
      elseif x >= vim.fn.strdisplaywidth(line)-7 then view.width=view.width+8; view.height=view.height+3; vim.schedule(position)
      elseif x >= vim.fn.strdisplaywidth(line)-11 then view.width=view.width-8; view.height=view.height-3; vim.schedule(position)
      else drag={row=pos.screenrow, col=pos.screencol, top=view.row, left=view.col} end
      return '<Ignore>'
    elseif key == '<LeftDrag>' and drag then
      view.row = drag.top + pos.screenrow - drag.row
      view.col = drag.left + pos.screencol - drag.col
      vim.schedule(position); return '<Ignore>'
    elseif key == '<LeftRelease>' and drag then
      drag=nil
      vim.schedule(function()
        if api.nvim_get_current_win() == hwin and vim.fn.mode():match('^[vV\22]') then
          api.nvim_feedkeys(api.nvim_replace_termcodes('<Esc>', true, false, true), 'n', false)
        end
      end)
      return '<Ignore>'
    end
    return key
  end
  for _, key in ipairs({'<LeftMouse>','<LeftDrag>','<LeftRelease>'}) do
    vim.keymap.set('n', key, function() return mouse(key) end, {buffer=header, expr=true, silent=true})
    vim.keymap.set('x', key, function() return mouse(key) end, {buffer=header, expr=true, silent=true})
  end
  vim.keymap.set('n', 'q', function() state.model_preview_hidden=true; require('model_maven.model_preview').close(state) end, {buffer=header,silent=true})
  for key, delta in pairs({h={0,-2},j={1,0},k={-1,0},l={0,2}}) do
    vim.keymap.set('n', key, function() view.row=view.row+delta[1]; view.col=view.col+delta[2]; position() end, {buffer=header,silent=true})
  end
  vim.keymap.set('n', '+', function() view.width=view.width+8; view.height=view.height+3; position() end, {buffer=header,silent=true})
  vim.keymap.set('n', '-', function() view.width=view.width-8; view.height=view.height-3; position() end, {buffer=header,silent=true})
  local group=api.nvim_create_augroup('ModelMavenLivePreview'..header, {clear=true})
  api.nvim_create_autocmd('VimResized', {group=group, callback=function()
    if vim.o.columns < 70 or vim.o.lines < 15 then
      vim.schedule(function()
        if not state.closed and state.model_preview == view then
          local preview = require('model_maven.model_preview')
          preview.close(state); preview.open(state)
        end
      end)
    else position() end
  end})
  api.nvim_create_autocmd('VimLeavePre', {group=group, once=true, callback=close})
  api.nvim_create_autocmd('BufEnter', {group=group, buffer=header, callback=function()
    local pos = vim.fn.getmousepos()
    if pos.winid == hwin then mouse('<LeftMouse>') end
  end})
  api.nvim_create_autocmd('BufEnter', {group=group, buffer=body, callback=function()
    if not view.closed then vim.schedule(function()
      if not view.closed and api.nvim_get_current_buf() == body then pcall(vim.cmd.startinsert) end
    end) end
  end})
  api.nvim_create_autocmd('BufWipeout', {group=group, buffer=header, once=true, callback=function()
    pcall(api.nvim_del_augroup_by_id, group)
    if state.model_preview == view then state.model_preview=nil; state.model_preview_hidden=true end
    close()
  end})
  local success, job = pcall(api.nvim_win_call, bwin, function()
    return vim.fn.jobstart({renderer, '--live-model-preview', path, '--lang', require('model_maven.locale').current()}, {term=true, on_exit=function(_, code)
      vim.schedule(function()
        if not view.closed and state.model_preview == view then
          state.model_preview=nil; close()
          if code ~= 0 then
            state.model_preview_live_unavailable = true
            require('model_maven.model_preview').open(state)
            vim.notify(tr('Interactive 3D is unavailable; showing a compact preview.', '交互 3D 不可用，已切换至紧凑预览。', '互動 3D 無法使用，已切換至緊湊預覽。'), vim.log.levels.WARN)
          else
            state.model_preview_hidden=true
            vim.notify(tr('3D preview closed.', '3D 预览已关闭。', '3D 預覽已關閉。'), vim.log.levels.INFO)
          end
        end
      end)
    end})
  end)
  if not success or type(job) ~= 'number' or job <= 0 then close(); return nil end
  view.job=job
  return view
end
return M
