local M = { workspace_api_version = 1 }
local defaults = { cli = vim.env.MODEL_MAVEN_CLI or 'm-model', python = nil }
local config = vim.deepcopy(defaults)
local tr = require('model_maven.locale').text

function M.setup(options)
  if options ~= nil and type(options) ~= 'table' then
    error('model_maven.setup() expects an options table', 2)
  end
  config = vim.tbl_deep_extend('force', config, options or {})
end

-- Health checks need the same configured executable as a normal plugin open.
-- Return a copy so callers cannot mutate the live configuration accidentally.
function M.get_config()
  return vim.deepcopy(config)
end

-- Resolve paths against the editor's current directory before the transport
-- switches its subprocess cwd to the project. Preserve executable symlinks,
-- which may be meaningful launchers or environment selectors.
function M.resolve_cli(value)
  if type(value) ~= 'string' or value == '' then return value end
  local resolved = value
  if not value:find('[/\\]') then
    local found = vim.fn.exepath(value)
    if found == '' then return value end
    resolved = found
  end
  return vim.fn.fnamemodify(resolved, ':p')
end

function M.open(options)
  if vim.fn.has('nvim-0.10') ~= 1 then
    vim.notify(tr('Model-Maven requires Neovim 0.10 or newer.', 'Model-Maven 需要 Neovim 0.10 或更新版本。', 'Model-Maven 需要 Neovim 0.10 或更新版本。'), vim.log.levels.ERROR)
    return
  end
  if options ~= nil and type(options) ~= 'table' then
    vim.notify(tr('Model-Maven options must be a table.', 'Model-Maven 选项必须是表。', 'Model-Maven 選項必須是表。'), vim.log.levels.ERROR)
    return
  end
  options = options or {}
  local root = vim.env.MODEL_MAVEN_PROJECT
  if options.project ~= nil then root = options.project end
  if root ~= nil and type(root) ~= 'string' then
    vim.notify(tr('Project must be a local directory path.', '项目必须是本地目录路径。', '專案必須是本機目錄路徑。'), vim.log.levels.ERROR)
    return
  end
  if not root or root == '' then
    local current = vim.api.nvim_buf_get_name(0)
    if current:match('^[%a][%w+.-]*://') then
      vim.notify(tr('The current buffer uses a remote or virtual URI. Open the workspace on that host, or explicitly choose a local project; no path mapping is assumed.',
        '当前 buffer 使用远程或虚拟 URI。请在对应主机打开工作区，或明确选择本地项目；不会自动假设路径映射。',
        '目前 buffer 使用遠端或虛擬 URI。請在對應主機開啟工作區，或明確選擇本機專案；不會自動假設路徑對應。'), vim.log.levels.WARN)
      return
    end
    root = vim.fs.root(current ~= '' and current or vim.fn.getcwd(), { '.model-maven', '.git' }) or vim.fn.getcwd()
  end
  if root:find('[%c]') then
    vim.notify(tr('Project path cannot contain control characters.', '项目路径不能包含控制字符。', '專案路徑不能包含控制字元。'), vim.log.levels.ERROR)
    return
  end
  if root:match('^[%a][%w+.-]*://') then
    vim.notify(tr('Project must be a directory visible to this Neovim process, not a URI.',
      '项目必须是当前 Neovim 进程可见的目录，不能是 URI。', '專案必須是目前 Neovim 程序可見的目錄，不能是 URI。'), vim.log.levels.WARN)
    return
  end
  root = vim.fn.fnamemodify(root, ':p')
  if vim.fn.isdirectory(root) ~= 1 then
    vim.notify(tr('Model-Maven: project directory does not exist: ', 'Model-Maven：项目目录不存在：', 'Model-Maven：專案目錄不存在：') .. root, vim.log.levels.ERROR)
    return
  end
  root = vim.uv.fs_realpath(root) or root
  local requested_cli = config.cli
  if options.cli ~= nil then requested_cli = options.cli end
  if type(requested_cli) ~= 'string' or requested_cli == '' or requested_cli:find('[%c]') then
    vim.notify(tr('Configure the Model-Maven CLI as a nonempty executable path or command.',
      '请将 Model-Maven CLI 配置为非空的可执行文件路径或命令。', '請將 Model-Maven CLI 設定為非空的執行檔路徑或命令。'), vim.log.levels.ERROR)
    return
  end
  local requested_python = config.python
  if options.python ~= nil then requested_python = options.python end
  if requested_python ~= nil and (type(requested_python) ~= 'string' or requested_python == ''
      or requested_python:find('[%c]')) then
    vim.notify(tr('Configure Python as a nonempty executable path or command, without control characters.',
      '请将 Python 配置为非空的可执行文件路径或命令，且不包含控制字符。',
      '請將 Python 設定為非空的執行檔路徑或命令，且不包含控制字元。'), vim.log.levels.ERROR)
    return
  end
  local unresolved_cli = type(requested_cli) == 'string' and requested_cli ~= ''
    and not requested_cli:find('[/\\]') and vim.fn.exepath(requested_cli) == ''
  require('model_maven.workspace').open({ project = root, cli = M.resolve_cli(requested_cli),
    cli_unresolved = unresolved_cli, python = requested_python,
    python_requested = options.python ~= nil or config.python ~= nil,
    open_empty_builder = options.open_empty_builder == true })
end

return M
