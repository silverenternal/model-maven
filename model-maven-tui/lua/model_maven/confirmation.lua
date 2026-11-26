-- Consequence-aware, revocable confirmation for workspace operations.
-- Callers must supply the complete reviewed scope and current selection. This
-- module deliberately does not infer paths, devices, environments, or effects.
local M = {}
local tr = require('model_maven.locale').text
local grants = {}

local function build_kinds()
  return {
  safe_read = {
    title = tr('Read only', '仅查看', '僅檢視'),
    effect = tr('No project files or model state will be changed.', '不会更改项目文件或模型状态。', '不會更改專案檔案或模型狀態。'),
    confirm = false,
  },
  user_code = {
    title = tr('Run user code', '执行用户代码', '執行使用者程式碼'),
    effect = tr('Selected project code will run in the selected runtime.', '将在所选运行环境中执行当前项目代码。', '將在所選執行階段中執行目前專案程式碼。'),
    confirm = true,
  },
  runtime_check = {
    title = tr('Check runtime', '检查运行环境', '檢查執行環境'),
    effect = tr('Starts the selected Python. Startup hooks and library initialization may run.',
      '将启动所选 Python，可能执行启动钩子和库初始化。', '將啟動所選 Python，可能執行啟動掛鉤和函式庫初始化。'),
    confirm = true,
    never_reuse = true,
  },
  persistent_write = {
    title = tr('Persistent change', '持久性修改', '持續性變更'),
    effect = tr('Files or project configuration may be created or changed.', '可能创建或修改文件及项目配置。', '可能建立或修改檔案及專案設定。'),
    confirm = true,
    never_reuse = true,
  },
  environment_change = {
    title = tr('Environment change', '环境变更', '環境變更'),
    effect = tr('The selected Python environment or its installed dependencies may change.',
      '所选 Python 环境或其中已安装的依赖可能发生变化。',
      '所選 Python 環境或其中已安裝的相依套件可能發生變更。'),
    confirm = true,
    never_reuse = true,
  },
  training = {
    title = tr('Training', '训练', '訓練'),
    effect = tr('Training may consume compute and persist checkpoints or run state.', '训练可能消耗计算资源并保存检查点或运行状态。', '訓練可能消耗運算資源並儲存檢查點或執行狀態。'),
    confirm = true,
    never_reuse = true,
  },
  unknown = {
    title = tr('Unclear outcome', '结果未知', '結果未知'),
    effect = tr('The outcome is not known; inspect the scope before continuing.', '操作后果尚不明确；请先检查范围再继续。', '操作後果尚不明確；請先檢查範圍再繼續。'),
    confirm = true,
    never_reuse = true,
    outcome_unknown = true,
  },
  }
end

local function encode(value)
  local ok, raw = pcall(vim.json.encode, value)
  if not ok or type(raw) ~= 'string' then return nil end
  return raw
end

local function has_fields(value)
  return type(value) == 'table' and next(value) ~= nil
end

local function fingerprint(spec)
  -- Scope and context are required. A missing identity must never collapse
  -- unrelated operations into a reusable session grant.
  -- Empty scope/context and an absent selection are not evidence of identity.
  -- The operation may still be confirmed once, but it cannot inherit a grant.
  if spec.identity_complete ~= true or not has_fields(spec.scope) or not has_fields(spec.context) or spec.selection == nil then return nil end
  local digest = spec.context.backend_identity_digest
  if type(digest) ~= 'string' or #digest ~= 64 or not digest:match('^[0-9a-f]+$') then return nil end
  return digest
end

local function matches(grant, reviewed)
  return grant and grant.kind == reviewed.kind and grant.effect == reviewed.effect
    and vim.deep_equal(grant.scope, reviewed.scope)
    and vim.deep_equal(grant.context, reviewed.context)
    and vim.deep_equal(grant.selection, reviewed.selection)
end

local function render(value)
  if value == nil then return tr('(none specified)', '（未提供）', '（未提供）') end
  if type(value) == 'string' then return value end
  local raw = encode(value)
  return raw or tr('(selection cannot be displayed)', '（无法显示所选内容）', '（無法顯示所選內容）')
end

local function present(spec, descriptor, callback, informational)
  local label = type(spec.label) == 'string' and spec.label or descriptor.title
  local effect = type(spec.effect) == 'string' and spec.effect or descriptor.effect
  local summary = tr('Action: ', '操作：', '操作：') .. label
    .. '\n' .. tr('Current selection: ', '当前选择：', '目前選擇：') .. render(spec.selection_summary or spec.selection)
    .. '\n' .. tr('Expected effect: ', '预期后果：', '預期後果：') .. effect
  if informational then
    summary = summary .. '\n' .. tr('Cannot continue until the outcome is known.', '结果明确前不能继续。', '結果明確前不能繼續。')
  end
  -- Selection providers may compress a multiline prompt to a single title.
  -- Keep the actual decision context in a scrollable native buffer instead.
  if vim.o.columns < 24 or vim.o.lines < 8 then
    vim.notify(tr('The terminal is too small to review this action.', '终端太小，无法审核此操作。', '終端太小，無法檢視此操作。'), vim.log.levels.WARN)
    callback(false, 'cancelled')
    return
  end
  local buf = vim.api.nvim_create_buf(false, true)
  local buttons = {}
  if not informational then
    buttons[#buttons + 1] = { key = 'a', label = tr('  → Continue [a]', '  → 继续 [a]', '  → 繼續 [a]'), approved = true }
    if spec.allow_edit == true then
      buttons[#buttons + 1] = { key = 'e', label = tr('  Edit selection [e]', '  修改选择 [e]', '  修改選擇 [e]'), reason = 'edit_selection' }
    end
  end
  buttons[#buttons + 1] = { key = 'q', label = informational
    and tr('  Close [q]', '  关闭 [q]', '  關閉 [q]')
    or tr('  ← Cancel [q]', '  ← 取消 [q]', '  ← 取消 [q]') }
  local lines = vim.tbl_map(function(button) return button.label end, buttons)
  lines[#lines + 1] = ''
  vim.list_extend(lines, vim.split(summary, '\n', { plain = true }))
  vim.api.nvim_buf_set_lines(buf, 0, -1, false, lines)
  vim.bo[buf].modifiable = false
  vim.bo[buf].bufhidden = 'wipe'
  local width, col = require('model_maven.model_preview').layout(100, function()
    for _, window in ipairs(vim.api.nvim_tabpage_list_wins(0)) do
      if vim.api.nvim_win_is_valid(window) and vim.b[vim.api.nvim_win_get_buf(window)].model_maven_preview then
        return vim.api.nvim_win_get_position(window)[2]
      end
    end
  end)
  local height = math.min(math.max(#lines, 6), vim.o.lines - 4)
  local ok, win = pcall(vim.api.nvim_open_win, buf, true, {
    relative = 'editor', style = 'minimal', border = 'single',
    width = width, height = height,
    row = math.floor((vim.o.lines - height) / 2) - 1,
    col = col,
  })
  if not ok then
    pcall(vim.api.nvim_buf_delete, buf, { force = true })
    callback(false, 'cancelled')
    return
  end
  vim.wo[win].wrap = true
  vim.wo[win].linebreak = true
  local finished = false
  local function finish(approved, reason)
    if finished then return end
    finished = true
    if vim.api.nvim_win_is_valid(win) then pcall(vim.api.nvim_win_close, win, true) end
    callback(approved, reason or (approved and 'confirmed' or 'cancelled'))
  end
  local function activate(button) finish(button.approved == true, button.reason) end
  local namespace = vim.api.nvim_create_namespace('ModelMavenConfirmation')
  for row, button in ipairs(buttons) do
    vim.api.nvim_buf_set_extmark(buf, namespace, row - 1, 0, {
      line_hl_group = button.approved and 'PmenuSel' or 'Pmenu',
    })
    vim.keymap.set('n', button.key, function() activate(button) end, { buffer = buf, silent = true })
  end
  if informational then vim.keymap.set('n', 'a', '<Nop>', { buffer = buf, silent = true }) end
  vim.keymap.set('n', '<CR>', function()
    local button = buttons[vim.api.nvim_win_get_cursor(win)[1]]
    if button then activate(button) end
  end, { buffer = buf, silent = true })
  vim.keymap.set('n', '<LeftMouse>', function()
    local mouse = vim.fn.getmousepos()
    local button = buttons[mouse.line]
    if mouse.winid ~= win or not button or mouse.column > #button.label then return '<LeftMouse>' end
    vim.schedule(function() if vim.api.nvim_win_is_valid(win) then activate(button) end end)
    return '<Ignore>'
  end, { buffer = buf, expr = true, silent = true })
  vim.keymap.set('n', '<Esc>', function() finish(false) end, { buffer = buf, silent = true })
  vim.api.nvim_create_autocmd('BufWipeout', { buffer = buf, once = true, callback = function() finish(false) end })
end

--- Request authorization for a described action.
--- spec.kind: safe_read | user_code | runtime_check | persistent_write | environment_change | training | unknown
--- spec.scope/context: complete caller-owned identity tables; include all
--- inputs whose change must invalidate an earlier decision.
--- opts.reuse_confirmed must be true to reuse an exact in-memory decision.
--- opts.edit_selection(spec, done) may present/edit the live selection before
--- review; the edited selection is fingerprinted and displayed for approval.
function M.request(spec, opts, callback)
  opts, callback = opts or {}, callback or function() end
  local kinds = build_kinds()
  if type(spec) ~= 'table' or not kinds[spec.kind] then
    callback(false, 'invalid_contract')
    return
  end
  local descriptor = kinds[spec.kind]
  local reviewed = vim.deepcopy(spec)
  local function review()
    -- An unclassifiable consequence cannot authorize a write or execution.
    -- The caller must resolve the state and make a newly classified request.
    if descriptor.outcome_unknown then
      present(reviewed, descriptor, function()
        callback(false, 'unknown_outcome', reviewed)
      end, true)
      return
    end
    if not descriptor.confirm then callback(true, 'read_only', reviewed); return end
    local key = fingerprint(reviewed)
    if opts.reuse_confirmed == true and not descriptor.never_reuse and key and matches(grants[key], reviewed) then
      callback(true, 'reused', reviewed)
      return
    end
    present(reviewed, descriptor, function(approved, result)
      if approved and opts.reuse_confirmed == true and not descriptor.never_reuse and key then
        grants[key] = vim.deepcopy(reviewed)
      end
      callback(approved, result, reviewed)
    end)
  end
  if type(opts.edit_selection) == 'function' then
    opts.edit_selection(reviewed, function(selection, accepted)
      if accepted ~= true then callback(false, 'cancelled', reviewed); return end
      reviewed.selection = selection
      review()
    end)
  else
    review()
  end
end

--- Revoke a single exact grant, or every grant when no scope is supplied.
function M.revoke(spec)
  if spec == nil then grants = {}; return true end
  local kinds = build_kinds()
  if type(spec) ~= 'table' or not kinds[spec.kind] then return false end
  local key = fingerprint(spec)
  if not key or not matches(grants[key], spec) then return false end
  grants[key] = nil
  return true
end

--- Invalidate grants whose caller-supplied context no longer matches.
--- Contexts should include project/session identity and relevant revisions.
function M.invalidate_context(context)
  if type(context) ~= 'table' then grants = {}; return end
  for key, grant in pairs(grants) do
    if not vim.deep_equal(grant.context, context) then grants[key] = nil end
  end
end

--- An unacknowledged or otherwise unknown operation outcome invalidates the
--- matching grant so a retry cannot inherit permission for an uncertain run.
function M.outcome_unknown(spec)
  return M.revoke(spec)
end

return M
