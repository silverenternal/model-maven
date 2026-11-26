-- Progress display only: never infer completion or percentages from time.
local M, api, uv = {}, vim.api, vim.uv
local tr = require('model_maven.locale').text
local frames = {'⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏'}
local function reduced() return vim.env.MODEL_MAVEN_REDUCE_MOTION == '1' end

function M.bar(step, total, width, active)
  width = math.max(8, math.min(30, math.floor(tonumber(width) or 20)))
  local moving = active == true and not reduced()
  local tick = moving and math.floor(uv.hrtime() / 120000000) or 0
  local icon = moving and frames[tick % #frames + 1] or (active and '•' or ' ')
  if type(step) == 'number' and type(total) == 'number' and step == step and total == total
      and step >= 0 and total > 0 and step <= total and total < math.huge then
    local count = math.floor(width * step / total)
    return icon .. '  ' .. string.rep('━', count) .. string.rep('╌', width - count)
      .. '  ' .. string.format('%d%%', math.floor(100 * step / total))
  end
  if not moving then
    return icon .. '  ' .. string.rep('╌', width) .. '  ' .. tr('progress unknown', '进度未知', '進度未知')
  end
  local cycle = math.max(1, (width - 3) * 2)
  local phase = tick % cycle
  local position = phase <= width - 3 and phase or cycle - phase
  return icon .. '  ' .. string.rep('╌', position) .. '━━━' .. string.rep('╌', width - position - 3)
    .. '  ' .. tr('progress unknown', '进度未知', '進度未知')
end

-- A non-focusable overlay stays visible over the graph during a CLI operation.
-- The timer changes only two short lines and never controls the actual process.
function M.update(state)
  local widget = state.progress_widget
  local function stop()
    if not widget then return end
    if widget.timer and not widget.timer:is_closing() then widget.timer:stop(); widget.timer:close() end
    if widget.win and api.nvim_win_is_valid(widget.win) then pcall(api.nvim_win_close, widget.win, true) end
    if widget.buf and api.nvim_buf_is_valid(widget.buf) then pcall(api.nvim_buf_delete, widget.buf, {force = true}) end
    if state.progress_widget == widget then state.progress_widget = nil end
  end
  if not state.busy or state.closed or not api.nvim_buf_is_valid(state.panel) then stop(); return end
  local function draw()
    if state.progress_widget ~= widget then return end
    if not state.busy or state.closed or not api.nvim_buf_is_valid(state.panel) then stop(); return end
    if not widget or not api.nvim_buf_is_valid(widget.buf) or not api.nvim_win_is_valid(widget.win) then stop(); return end
    local width = math.max(20, math.min(58, vim.o.columns - 4))
    local label = tostring(state.busy):gsub('[%c]', ' ')
    label = vim.fn.strcharpart(label, 0, 48)
    local title = (state.cancel_requested and tr('Stopping · ', '正在停止 · ', '正在停止 · ') or tr('Working · ', '执行中 · ', '執行中 · ')) .. label
    while vim.fn.strdisplaywidth(title) > width do title = vim.fn.strcharpart(title, 0, vim.fn.strchars(title) - 1) end
    local bar = M.bar(nil, nil, math.max(8, width - 21), true)
    pcall(api.nvim_win_set_config, widget.win, {relative='editor', row=0,
      col=math.max(0, math.floor((vim.o.columns-width)/2)), width=width, height=2})
    vim.bo[widget.buf].modifiable = true
    api.nvim_buf_set_lines(widget.buf, 0, -1, false, {title, bar})
    vim.bo[widget.buf].modifiable = false
  end
  if not widget then
    if vim.o.columns < 24 or vim.o.lines < 6 then return end
    local buf = api.nvim_create_buf(false, true)
    vim.bo[buf].buftype, vim.bo[buf].bufhidden, vim.bo[buf].swapfile = 'nofile', 'wipe', false
    local width = math.max(20, math.min(58, vim.o.columns - 4))
    local ok, win = pcall(api.nvim_open_win, buf, false, {relative='editor', style='minimal', border='rounded',
      row=0, col=math.max(0, math.floor((vim.o.columns-width)/2)), width=width, height=2, focusable=false, zindex=250, noautocmd=true})
    if not ok then pcall(api.nvim_buf_delete, buf, {force=true}); return end
    widget = {buf=buf, win=win}; state.progress_widget = widget
    vim.wo[win].wrap = false
    if not reduced() then
      widget.timer = uv.new_timer()
      if widget.timer then widget.timer:start(120, 120, vim.schedule_wrap(draw)) end
    end
  end
  draw()
end

return M
