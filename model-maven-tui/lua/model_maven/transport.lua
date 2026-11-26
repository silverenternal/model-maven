-- The CLI remains the authority for execution, reports and persistence.
local M = {}
local tr = require('model_maven.locale').text
local LIMIT = 32 * 1024 * 1024

local function diagnostic(message, stderr)
  return { kind = 'transport_error', message = message, stderr = stderr or '', outcome_unknown = true, operation_started = true }
end

local function not_started(message, stderr)
  local result = diagnostic(message, stderr)
  result.operation_started = false
  result.outcome_unknown = false
  return result
end

local function redact_stderr(stderr, environment)
  local text = tostring(stderr or '')
  for name, value in pairs(environment or {}) do
    local key = type(name) == 'string' and name:lower() or ''
    local sensitive = key:find('token', 1, true) or key:find('secret', 1, true)
      or key:find('password', 1, true) or key:find('credential', 1, true)
      or key:find('key', 1, true) or key:find('auth', 1, true) or key:find('cookie', 1, true)
    if sensitive and type(value) == 'string' and value ~= '' then
      text = text:gsub(vim.pesc(value), '[redacted]')
    end
  end
  return text
end

local function is_uri(value)
  if type(value) ~= 'string' then return false end
  -- Keep Windows drive paths local while rejecting URI schemes such as
  -- zipfile:/... that do not use the common :// spelling.
  if vim.fn.has('win32') == 1 and value:match('^[%a]:') then return false end
  return value:match('^[%a][%w+.-]*:') ~= nil
end

function M.request(opts, args, callback, preview_paths_checked)
  -- vim.system executes in the editor's local process namespace. A URI is not
  -- a local cwd and must never be guessed into a host path or remote mapping.
  if is_uri(opts.project) then
    vim.schedule(function()
      local unsupported = not_started(tr('The selected project is a remote or virtual URI; this transport has no trusted path mapping. Open the workspace on that host or select a local project.',
        '所选项目是远程或虚拟 URI；当前传输没有可信路径映射。请在对应主机打开工作区或选择本地项目。',
        '所選專案是遠端或虛擬 URI；目前傳輸沒有可信路徑對映。請在對應主機開啟工作區或選擇本機專案。'))
      unsupported.kind = 'path_mapping_unavailable'
      callback(nil, unsupported)
    end)
    return nil
  end
  -- A bare command absent from the editor's PATH must not be searched again
  -- after changing cwd to the project: relative PATH entries could select a
  -- different executable there. Explicit session selection resolves it in the editor environment.
  if opts.cli_unresolved then
    vim.schedule(function()
      callback(nil, not_started(tr('CLI was not found in the editor environment: ', '编辑器环境中找不到 CLI：', '編輯器環境中找不到 CLI：')
        .. tostring(opts.cli) .. tr('. Install it or update PATH, then choose it from the workspace environment menu.',
          '。请安装或更新 PATH，然后在工作区环境菜单中选择它。', '。請安裝或更新 PATH，然後在工作區環境選單中選擇它。')))
    end)
    return nil
  end
  local has_paths = opts.runtime_paths_override == true or opts.working_directory ~= nil or (type(opts.project_import_roots) == 'table' and #opts.project_import_roots > 0)
  local data_preview = args[1] == 'data' and args[2] == 'preview'
  local assembly_generate = args[1] == 'assembly' and args[2] == 'generate'
  local assembly_show = args[1] == 'assembly' and args[2] == 'show' and not vim.tbl_contains(args, '--config-only')
  local generic_trial = args[1] == 'loop-trial'
  local generic_run = args[1] == 'loop-run'
  local authorization_identity = args[1] == 'workspace'
    and (args[2] == 'authorization-identity' or args[2] == 'run')
  local paths_apply = args[1] == 'graph-draft-probe' or args[1] == 'model' or args[1] == 'layer' or args[1] == 'try'
    or args[1] == 'train' or args[1] == 'evaluate' or (args[1] == 'env' and (args[2] == 'probe' or args[2] == 'devices'))
    or data_preview or generic_trial or generic_run
    or (args[1] == 'assembly' and args[2] == 'patch') or assembly_generate or assembly_show
    or authorization_identity
  if has_paths and paths_apply and not preview_paths_checked then
    local selected = { project = opts.project, python = opts.python, cli = opts.cli,
      working_directory = opts.working_directory, project_import_roots = vim.deepcopy(opts.project_import_roots),
      runtime_paths_override = opts.runtime_paths_override }
    -- Keep one cancellation handle across capability discovery and the real
    -- request. A cancellation during discovery must never start user code.
    local handle = { phase = 'checking', cancelled = false }
    function handle:kill(signal)
      if self.phase == 'checking' then
        self.cancelled = true
        if self.current then pcall(self.current.kill, self.current, signal) end
        return true
      end
      if self.phase == 'running' and self.current then
        return self.current:kill(signal)
      end
      return false, 'process unavailable'
    end
    handle.current = M.request(opts, { '--describe-interface' }, function(report, err, stderr)
      if handle.cancelled then
        handle.phase = 'done'
        local cancelled = not_started(tr('Cancelled during capability discovery; the requested operation did not start.',
          '能力检查期间已取消；请求的操作尚未启动。', '能力檢查期間已取消；要求的操作尚未啟動。'))
        cancelled.kind = 'cancelled_before_start'
        return callback(nil, cancelled)
      end
      local caps = report and report.operation == 'interface' and report.status == 'success'
        and report.data and report.data.capabilities
      local required = generic_trial and 'training.generic_loop_trial_cli_v1'
        or generic_run and 'training.generic_loop_train_cli_v1'
        or data_preview and 'runtime.data_preview_paths_v1'
        or (assembly_generate and 'assembly.generate_runtime_paths_v1'
          or (assembly_show and 'assembly.show_runtime_paths_v1' or nil))
      if err then
        handle.phase = 'done'
        local reason = type(err.message) == 'string' and vim.fn.strcharpart(err.message:gsub('[%c]', ' '), 0, 500)
          or tr('Capability discovery failed.', '能力检查失败。', '能力檢查失敗。')
        return callback(nil, not_started(tr('The requested operation did not start because CLI capability discovery failed: ',
          '请求的操作尚未启动，因为 CLI 能力检查失败：', '要求的操作尚未啟動，因為 CLI 能力檢查失敗：') .. reason, stderr))
      end
      if type(caps) ~= 'table' or caps['runtime.workspace_paths_v1'] ~= true
          or (required and caps[required] ~= true) then
        handle.phase = 'done'
        return callback(nil, not_started(tr('This CLI cannot guarantee the selected runtime paths for this action.',
          '此 CLI 无法保证当前操作使用所选运行时路径。', '此 CLI 無法保證目前操作使用所選執行階段路徑。'), stderr))
      end
      if opts.project ~= selected.project or opts.python ~= selected.python or opts.cli ~= selected.cli
          or opts.working_directory ~= selected.working_directory or opts.runtime_paths_override ~= selected.runtime_paths_override
          or not vim.deep_equal(opts.project_import_roots, selected.project_import_roots) then
        handle.phase = 'done'
        return callback(nil, not_started(tr('Runtime context changed; request the preview again.',
          '运行时上下文已变化，请重新请求操作。', '執行階段上下文已變更，請重新請求操作。')))
      end
      handle.phase = 'running'
      handle.current = M.request(opts, args, function(...)
        handle.phase = 'done'
        callback(...)
      end, true)
    end)
    return handle
  end
  local metadata_only = args[1] == '--describe-interface' or args[1] == 'discover'
    or args[1] == 'runs' or args[1] == 'variant' or args[1] == 'export' or (args[1] == 'assembly' and args[2] == 'show' and vim.tbl_contains(args, '--config-only'))
    -- These operations plan or write project files without importing user
    -- model code; they must not consume the runtime's working directory.
    or args[1] == 'init' or args[1] == 'config' or args[1] == 'scaffold'
    or args[1] == 'recipe-update' or args[1] == 'recipe-bundle' or args[1] == 'recipe-sync'
    or (args[1] == 'data' and args[2] == 'csv-template')
    or (args[1] == 'env' and (args[2] == 'check' or args[2] == 'libraries'))
  if has_paths and not paths_apply and not metadata_only then
    vim.schedule(function()
      callback(nil, not_started(tr('This action does not yet support the selected runtime paths. Reset them or use a supported model action.',
        '此操作尚不支持所选运行时路径，请重置路径或使用已支持的模型操作。',
        '此操作尚不支援所選執行階段路徑，請重設路徑或使用已支援的模型操作。')))
    end)
    return nil
  end
  if opts.before_process and args[1] ~= '--describe-interface' then
    local ok, current = pcall(opts.before_process)
    if not ok or not current then
      vim.schedule(function()
        local changed = not_started(tr('Context changed before the operation started; review and try again.',
          '操作启动前上下文已变化，请重新检查后再试。', '操作啟動前上下文已變更，請重新檢查後再試。'))
        changed.kind = 'context_changed_before_start'
        callback(nil, changed)
      end)
      return nil
    end
  end
  local command = { opts.cli or 'm-model', '--json', '--no-interactive' }
  if opts.python and opts.python ~= '' then
    vim.list_extend(command, { '--python', opts.python })
  end
  vim.list_extend(command, args)
  local output, errors, bytes, too_large, stream_error = {}, {}, 0, false, nil
  local process
  local function collect(target, err, data)
    if err then stream_error = tostring(err) end
    if not data then return end
    bytes = bytes + #data
    if bytes > LIMIT then
      too_large = true
      -- Drain output without terminating the CLI: it may be committing a write.
      return
    end
    target[#target + 1] = data
  end
  local environment = vim.fn.environ()
  environment.MODEL_MAVEN_WORKSPACE_WORKING_DIRECTORY = nil
  environment.MODEL_MAVEN_WORKSPACE_IMPORT_ROOTS = nil
  environment.MODEL_MAVEN_FORCE_FRESH_FORWARD = opts.fresh_forward and '1' or nil
  if paths_apply then
    environment.MODEL_MAVEN_WORKSPACE_WORKING_DIRECTORY = opts.working_directory or (opts.runtime_paths_override and opts.project or nil)
    if type(opts.project_import_roots) == 'table' and #opts.project_import_roots > 0 then
      environment.MODEL_MAVEN_WORKSPACE_IMPORT_ROOTS = vim.json.encode(opts.project_import_roots)
    elseif opts.runtime_paths_override then
      environment.MODEL_MAVEN_WORKSPACE_IMPORT_ROOTS = '[]'
    end
  end
  -- Before Neovim 0.11.3, clear_env passes env straight to uv.spawn;
  -- uv expects an array of KEY=VALUE strings, not a keyed table. Newer
  -- Neovim versions convert the keyed form themselves.
  local spawn_environment = environment
  if vim.fn.has('nvim-0.11.3') == 0 then
    spawn_environment = {}
    for key, value in pairs(environment) do
      if type(key) == 'string' and type(value) == 'string' then
        spawn_environment[#spawn_environment + 1] = key .. '=' .. value
      end
    end
    table.sort(spawn_environment)
  end
  local ok, result = pcall(vim.system, command, {
    cwd = opts.project, text = true,
    env = spawn_environment, clear_env = true,
    stdout = function(err, data) collect(output, err, data) end,
    stderr = function(err, data) collect(errors, err, data) end,
  }, function(exit)
    vim.schedule(function()
      local stderr = redact_stderr(table.concat(errors), environment)
      if exit.signal and exit.signal ~= 0 then
        local interrupted = diagnostic(tr('CLI transport ended after the process started; execution state is unknown. Query the existing run or request before retrying.',
          'CLI 传输在进程启动后中断；执行状态未知。重试前请查询现有运行或请求。',
          'CLI 傳輸在程序啟動後中斷；執行狀態未知。重試前請查詢現有執行或要求。'), stderr)
        interrupted.kind = 'transport_interrupted'
        interrupted.operation_started = true
        return callback(nil, interrupted, stderr)
      end
      if too_large then return callback(nil, diagnostic(tr('CLI output exceeded 32 MiB. The process exited, but its result is unknown; inspect project state before retrying.', 'CLI 输出超过 32 MiB。进程已退出但结果未知；重试前请检查项目状态。', 'CLI 輸出超過 32 MiB。程序已退出但結果未知；重試前請檢查專案狀態。'), stderr)) end
      if stream_error then return callback(nil, diagnostic(tr('Could not read the complete CLI output; inspect the outcome before retrying: ', '无法完整读取 CLI 输出；重试前请检查操作结果：', '無法完整讀取 CLI 輸出；重試前請檢查操作結果：') .. redact_stderr(stream_error, environment), stderr)) end
      local raw = table.concat(output)
      local decoded, report = pcall(vim.json.decode, raw)
      if not decoded or type(report) ~= 'table' or report.protocol_version ~= 1
          or type(report.operation) ~= 'string' or report.operation == ''
          or (report.status ~= 'success' and report.status ~= 'error' and report.status ~= 'inconclusive') then
        local invalid = diagnostic(tr('CLI returned an invalid report after the process started; execution state is unknown. Inspect project state before retrying. Exit code: ', 'CLI 启动后返回了无效报告，执行状态未知。重试前请检查项目状态。退出码：', 'CLI 啟動後回傳無效報告，執行狀態未知。重試前請檢查專案狀態。結束碼：') .. tostring(exit.code), stderr)
        invalid.operation_started = true
        return callback(nil, invalid, stderr)
      end
      if (exit.code == 0) ~= (report.status == 'success') then
        return callback(nil, diagnostic(tr('CLI exit status does not match the report.', 'CLI 退出状态与报告不一致。', 'CLI 結束狀態與報告不一致。'), stderr))
      end
      if report.status == 'success' and (report.data == nil or report.data == vim.NIL) then
        return callback(nil, diagnostic(tr('Successful CLI report has no data.', 'CLI 成功报告缺少数据。', 'CLI 成功報告缺少資料。'), stderr))
      end
      if report.operation == 'interface' and report.status == 'success' then
        local data = report.data
        if type(data) ~= 'table' or data.interface_version ~= 1 or data.report_protocol_version ~= 1
            or type(data.cli_version) ~= 'string' or data.cli_version == '' or type(data.capabilities) ~= 'table' then
          return callback(nil, diagnostic(tr('CLI capability catalog is incompatible or invalid; guarded actions are unavailable.',
            'CLI 能力目录不兼容或无效；受保护操作不可用。', 'CLI 能力目錄不相容或無效；受保護操作無法使用。'), stderr))
        end
        for name, advertised in pairs(data.capabilities) do
          if type(name) ~= 'string' or name == '' or type(advertised) ~= 'boolean' then
            return callback(nil, diagnostic(tr('CLI capability catalog contains an invalid entry; guarded actions are unavailable.',
              'CLI 能力目录包含无效条目；受保护操作不可用。', 'CLI 能力目錄包含無效項目；受保護操作無法使用。'), stderr))
          end
        end
      end
      callback(report, nil, stderr)
    end)
  end)
  if not ok then
    vim.schedule(function() callback(nil, not_started(tr('Could not start m-model: ', '无法启动 m-model：', '無法啟動 m-model：') .. tostring(result))) end)
    return nil
  end
  process = result
  return process
end

return M
