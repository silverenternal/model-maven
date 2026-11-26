-- A temporary, project-session input buffer. Nothing here writes project files.
local M = {}
local api = vim.api

function M.open(store, key, title, default, baseline, submit, tr)
  local entry = store[key]
  local stale = entry and not vim.deep_equal(entry.baseline, baseline)
  if not entry or not api.nvim_buf_is_valid(entry.buf) then
    local buf = api.nvim_create_buf(false, true)
    vim.bo[buf].buftype, vim.bo[buf].bufhidden, vim.bo[buf].swapfile = 'nofile', 'hide', false
    api.nvim_buf_set_lines(buf, 0, -1, false, vim.split(default or '', '\n', {plain = true}))
    entry = {buf = buf, baseline = vim.deepcopy(baseline)}
    store[key] = entry
    stale = false
  end
  local buf = entry.buf
  local existing = vim.fn.win_findbuf(buf)
  local win
  if #existing > 0 and api.nvim_win_is_valid(existing[1]) then
    win = existing[1]
    api.nvim_set_current_win(win)
  else
    local width, height = math.min(100, vim.o.columns - 4), math.min(10, vim.o.lines - 5)
    if width < 20 or height < 3 then return vim.notify(tr('Enlarge the terminal to edit this draft.', '请放大终端以编辑草稿。', '請放大終端以編輯草稿。'), vim.log.levels.WARN) end
    local ok, result = pcall(api.nvim_open_win, buf, true, {relative = 'editor', style = 'minimal', border = 'rounded',
      width = width, height = height, row = 2, col = math.floor((vim.o.columns - width) / 2), zindex = 200})
    if not ok then return vim.notify(tostring(result), vim.log.levels.ERROR) end
    win = result
  end
  vim.wo[win].winbar = (title .. tr('  Ctrl-S apply  Ctrl-Q hide  Ctrl-X discard', '  Ctrl-S 应用  Ctrl-Q 暂藏  Ctrl-X 丢弃', '  Ctrl-S 套用  Ctrl-Q 暫藏  Ctrl-X 捨棄')):gsub('%%', '%%%%')
  if entry.cursor then pcall(api.nvim_win_set_cursor, win, entry.cursor) end
  local function hide()
    if api.nvim_win_is_valid(win) and api.nvim_win_get_buf(win) == buf then
      entry.cursor = api.nvim_win_get_cursor(win)
      pcall(api.nvim_win_close, win, false)
    end
  end
  local function discard()
    store[key] = nil
    hide()
    if api.nvim_buf_is_valid(buf) then pcall(api.nvim_buf_delete, buf, {force = true}) end
  end
  local function apply()
    if not api.nvim_buf_is_valid(buf) or entry.applying then return end
    if stale then return vim.notify(tr('The graph changed; copy or discard this draft before editing the new graph.',
      '图已变化，请先复制或丢弃此草稿，再编辑新图。', '圖已變更，請先複製或捨棄此草稿，再編輯新圖。'), vim.log.levels.WARN) end
    local raw = table.concat(api.nvim_buf_get_lines(buf, 0, -1, false), '\n')
    if #raw > 65536 then return vim.notify(tr('Input exceeds 64 KiB.', '输入超过 64 KiB。', '輸入超過 64 KiB。'), vim.log.levels.WARN) end
    entry.applying = true
    local tick = api.nvim_buf_get_changedtick(buf)
    local function completed()
      entry.applying = false
      if store[key] ~= entry or not api.nvim_buf_is_valid(buf) then return end
      if api.nvim_buf_get_changedtick(buf) == tick then discard()
      else vim.notify(tr('Applied the earlier input; newer edits remain as a draft.', '已应用先前输入；较新的编辑仍保留为草稿。', '已套用先前輸入；較新的編輯仍保留為草稿。'), vim.log.levels.WARN) end
    end
    local started, reason = submit(raw, completed)
    entry.applying = false
    if not started then
      entry.applying = false
      if reason then vim.notify(reason, vim.log.levels.WARN) end
    end
  end
  for _, mode in ipairs({'n', 'i'}) do
    vim.keymap.set(mode, '<C-s>', apply, {buffer = buf, silent = true, desc = 'Apply parameter draft'})
    vim.keymap.set(mode, '<C-q>', hide, {buffer = buf, silent = true, desc = 'Hide parameter draft'})
    vim.keymap.set(mode, '<C-x>', discard, {buffer = buf, silent = true, desc = 'Discard parameter draft'})
  end
  vim.keymap.set('n', 'q', hide, {buffer = buf, silent = true, desc = 'Hide parameter draft'})
  vim.keymap.set('n', '<CR>', apply, {buffer = buf, silent = true, desc = 'Apply parameter draft'})
  vim.notify(title .. tr(' · Ctrl-S apply · Ctrl-Q hide · Ctrl-X discard (session only)',
    ' · Ctrl-S 应用 · Ctrl-Q 暂藏 · Ctrl-X 丢弃（仅本会话）',
    ' · Ctrl-S 套用 · Ctrl-Q 暫藏 · Ctrl-X 捨棄（僅本工作階段）'), vim.log.levels.INFO)
end
return M
