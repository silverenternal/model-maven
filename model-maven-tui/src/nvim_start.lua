-- Bootstrap the standalone workspace without altering user mappings. The
-- session theme is neutralized only when NO_COLOR is explicitly set.
local function nonempty(value) return value ~= nil and value ~= '' and value or nil end
local preferred = nonempty(vim.env.MODEL_MAVEN_LANGUAGE)
if preferred and preferred:lower() == 'system' then preferred = nil end
local language = (preferred or nonempty(vim.env.LC_ALL) or nonempty(vim.env.LC_MESSAGES) or nonempty(vim.env.LANG) or ''):lower():gsub('_', '-')
local traditional = language:find('hant', 1, true) or language:find('zh-tw', 1, true)
  or language:find('zh-hk', 1, true) or language:find('zh-mo', 1, true)
local function tr(en, zh, tw)
  if traditional then return tw end
  if language:match('^zh') then return zh end
  return en
end
if vim.fn.has('nvim-0.10') ~= 1 then
  vim.notify(tr('Model-Maven requires Neovim 0.10 or newer. Upgrade Neovim or set MODEL_MAVEN_NVIM to a compatible executable.',
    'Model-Maven 需要 Neovim 0.10 或更新版本。请升级 Neovim 或设置 MODEL_MAVEN_NVIM。',
    'Model-Maven 需要 Neovim 0.10 或更新版本。請升級 Neovim 或設定 MODEL_MAVEN_NVIM。'), vim.log.levels.ERROR)
  return
end
if vim.v.errmsg ~= '' then
  vim.notify(tr('Neovim reported a startup error, possibly from init.lua or a plugin. Inspect :messages.',
    'Neovim 启动时报告了错误，可能来自配置或插件。请查看 :messages。',
    'Neovim 啟動時回報了錯誤，可能來自設定或外掛。請查看 :messages。'), vim.log.levels.WARN)
end
vim.opt.runtimepath:prepend(vim.env.MODEL_MAVEN_RUNTIME_DIR)
local loaded, load_error = pcall(vim.cmd, 'runtime plugin/model_maven.lua')
if not loaded then
  vim.notify(tr('Could not load the Model-Maven plugin. Check plugin files and runtimepath: ',
    '无法加载 Model-Maven 插件，请检查插件文件和 runtimepath：',
    '無法載入 Model-Maven 外掛，請檢查外掛檔案和 runtimepath：') .. tostring(load_error), vim.log.levels.ERROR)
  return
end
local required, workspace = pcall(require, 'model_maven')
if not required then
  vim.notify(tr('Could not load the Model-Maven workspace. Check :messages and runtimepath: ',
    '无法加载 Model-Maven 工作区，请检查 :messages 和 runtimepath：',
    '無法載入 Model-Maven 工作區，請檢查 :messages 和 runtimepath：') .. tostring(workspace), vim.log.levels.ERROR)
  return
end
local info = type(workspace) == 'table' and type(workspace.open) == 'function' and debug.getinfo(workspace.open, 'S')
local source = info and info.source or ''
local expected = vim.fs.normalize(vim.env.MODEL_MAVEN_RUNTIME_DIR .. '/lua/model_maven/init.lua')
if type(workspace) == 'table' and workspace.workspace_api_version == 1 and source:sub(1, 1) == '@'
    and vim.fs.normalize(source:sub(2)) == expected then
  workspace.open({cli = vim.env.MODEL_MAVEN_CLI, open_empty_builder = true})
  if vim.env.NO_COLOR ~= nil then
    require('model_maven.no_color').setup()
  end
else
  vim.notify(tr('A different Model-Maven plugin is already loaded. Disable the conflicting plugin or use :ModelMaven.',
    '已加载另一个 Model-Maven 插件。请禁用冲突插件或使用 :ModelMaven。',
    '已載入另一個 Model-Maven 外掛。請停用衝突外掛或使用 :ModelMaven。'), vim.log.levels.ERROR)
end
