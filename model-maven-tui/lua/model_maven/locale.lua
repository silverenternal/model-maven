local M = {}
local selected
local uv = vim.uv or vim.loop
-- Cache only the preference value; explicit session choice remains immediate.
-- Other processes' preference writes become visible after at most one second.
local preference_cache = { path = nil, language = 'system', refresh_at = 0 }
local labels = {
  system = 'Follow system / 跟随系统 / 跟隨系統',
  en = 'English', ['zh-cn'] = '简体中文', ['zh-tw'] = '繁體中文',
}

local function preference_path()
  if vim.env.MODEL_MAVEN_TUI_PREFERENCES and vim.env.MODEL_MAVEN_TUI_PREFERENCES ~= '' then
    return vim.env.MODEL_MAVEN_TUI_PREFERENCES
  end
  local root
  if vim.fn.has('win32') == 1 then
    root = vim.env.APPDATA
  elseif vim.fn.has('mac') == 1 then
    root = vim.env.HOME and (vim.env.HOME .. '/Library/Application Support')
  else
    root = vim.env.XDG_CONFIG_HOME
    if not root or root:sub(1, 1) ~= '/' then root = vim.env.HOME and (vim.env.HOME .. '/.config') end
  end
  return root and (root .. '/model-maven-tui/preferences.json') or nil
end

local function read_document()
  local path = preference_path()
  if not path then return {}, nil end
  local file = io.open(path, 'rb')
  if not file then
    if uv.fs_stat(path) then return nil, path end
    return {}, path, false
  end
  local raw = file:read(1024 * 1024 + 1) or ''
  file:close()
  if #raw > 1024 * 1024 then return nil, path end
  local ok, decoded = pcall(vim.json.decode, raw)
  if not ok or type(decoded) ~= 'table' or not raw:match('^%s*{') then return nil, path end
  return decoded, path, raw
end

local function configured_language(value)
  if type(value) ~= 'string' then return nil end
  local normalized = value:lower():gsub('_', '-'):gsub('%.utf%-8$', ''):gsub('%.utf8$', '')
  if normalized == 'system' then return 'system' end
  if normalized == 'en' or normalized:match('^en%-') then return 'en' end
  if normalized == 'zh' or normalized == 'zh-cn' or normalized == 'zh-sg' or normalized == 'zh-hans'
      or normalized:match('^zh%-hans%-') then return 'zh-cn' end
  if normalized == 'zh-tw' or normalized == 'zh-hk' or normalized == 'zh-mo'
      or normalized == 'zh-hant' or normalized:match('^zh%-hant%-') then return 'zh-tw' end
  return nil
end

local function resolve(value)
  local configured = configured_language(value)
  if configured and configured ~= 'system' then return configured end
  local tag = (vim.env.LC_ALL ~= '' and vim.env.LC_ALL) or (vim.env.LC_MESSAGES ~= '' and vim.env.LC_MESSAGES) or vim.env.LANG or ''
  tag = tag:lower():gsub('_', '-')
  if tag:match('^zh') then
    if tag:find('hant', 1, true) or tag:find('-tw', 1, true) or tag:find('-hk', 1, true) or tag:find('-mo', 1, true) then return 'zh-tw' end
    return 'zh-cn'
  end
  return 'en'
end

function M.current()
  if selected then return resolve(selected) end
  local environment_language = configured_language(vim.env.MODEL_MAVEN_LANGUAGE)
  if environment_language then return resolve(environment_language) end
  local path, now = preference_path(), uv.hrtime()
  if preference_cache.path ~= path or now >= preference_cache.refresh_at then
    local doc = read_document()
    preference_cache.path = path
    preference_cache.language = doc and doc.language or 'system'
    preference_cache.refresh_at = now + 1000000000
  end
  return resolve(preference_cache.language)
end

function M.text(en, zh, tw)
  local locale = M.current()
  return locale == 'zh-cn' and zh or (locale == 'zh-tw' and tw or en)
end

-- Translates workspace-owned messages; backend diagnostics remain intact.
local messages = {
  ['Cannot encode draft parameter.'] = { '无法编码草稿参数。', '無法編碼草稿參數。' },
  ['Cannot encode draft operation.'] = { '无法编码草稿操作。', '無法編碼草稿操作。' },
  ['Model preview is too large to approve.'] = { '模型预览过大，无法在此确认。', '模型預覽過大，無法在此確認。' },
  ['Cannot encode layer operation.'] = { '无法编码层操作。', '無法編碼層操作。' },
  ['No recorded module paths.'] = { '没有已记录的模块路径。', '沒有已記錄的模組路徑。' },
  ['No verifiable implementation identity was recorded.'] = { '没有记录可核对的实现身份。', '沒有記錄可核對的實作身分。' },
  ['Project directory is unavailable.'] = { '项目目录不可用。', '專案目錄無法使用。' },
  ['Model-Maven operation finished after its workspace closed; inspect project or run state before retrying.'] = { '工作区关闭后操作才结束；重试前请检查项目或运行状态。', '工作區關閉後操作才結束；重試前請檢查專案或執行狀態。' },
  ['This diagnostic is no longer current; run the forward pass again.'] = { '诊断已过期，请重新执行前向。', '診斷已過期，請重新執行前向。' },
  ['No modules reported.'] = { '报告中没有模块。', '報告中沒有模組。' },
  ['No project-source implementation with a content identity was reported.'] = { '报告未提供带内容身份的项目源码实现。', '報告未提供帶內容身分的專案原始碼實作。' },
  ['Implementation path cannot be verified within this project.'] = { '无法确认实现路径位于项目内。', '無法確認實作路徑位於專案內。' },
  ['Implementation is outside the project or is unavailable.'] = { '实现位于项目外或不可访问。', '實作位於專案外或無法存取。' },
  ['Implementation cannot be safely verified.'] = { '无法安全核对实现源码。', '無法安全核對實作原始碼。' },
  ['The source buffer has unsaved edits; save it and run forward again before locating.'] = { '源码有未保存修改；请保存并重新前向后再定位。', '原始碼有未儲存修改；請儲存並重新前向後再定位。' },
  ['Cannot read implementation source.'] = { '无法读取实现源码。', '無法讀取實作原始碼。' },
  ['Implementation changed since the forward pass; run it again before locating.'] = { '前向后源码已变化；请重新运行后再定位。', '前向後原始碼已變更；請重新執行後再定位。' },
  ['The open buffer encoding cannot be compared to the observed source.'] = { '无法将已打开的 buffer 编码与观察源码比较。', '無法將已開啟的 buffer 編碼與觀察原始碼比較。' },
  ['The open buffer differs from disk; reconcile it and run forward again.'] = { '已打开的 buffer 与磁盘不同；请先处理差异并重新前向。', '已開啟的 buffer 與磁碟不同；請先處理差異並重新前向。' },
  ['An open source buffer changed during verification; reconcile it before locating.'] = { '核对期间源码 buffer 已变化；请处理后再定位。', '核對期間原始碼 buffer 已變更；請處理後再定位。' },
  ['An open buffer encoding or line format cannot be compared to the observed source.'] = { '已打开 buffer 的编码或换行格式无法与观察源码比较。', '已開啟 buffer 的編碼或換行格式無法與觀察原始碼比較。' },
  ['An open buffer for this file differs from disk; reconcile it before locating.'] = { '该文件的已打开 buffer 与磁盘不同；请处理后再定位。', '該檔案的已開啟 buffer 與磁碟不同；請處理後再定位。' },
  ['Model-Maven: unsaved project buffers; save them or cancel before running.\n'] = { 'Model-Maven：项目有未保存的 buffer；请保存或取消。\n', 'Model-Maven：專案有未儲存的 buffer；請儲存或取消。\n' },
  ['Model-Maven: project buffers changed; save them before running.'] = { 'Model-Maven：项目 buffer 已变化；运行前请保存。', 'Model-Maven：專案 buffer 已變更；執行前請儲存。' },
  ['This CLI cannot safely pin and track a resume request.'] = { '当前 CLI 无法安全固定检查点并追踪续训请求。', '目前 CLI 無法安全固定檢查點並追蹤續訓要求。' },
  ['A verified checkpoint identity is required to resume.'] = { '续训需要可核对的检查点身份。', '續訓需要可核對的檢查點身分。' },
  ['Resolve pending training requests before resuming: '] = { '续训前请核查未确认的训练请求：', '續訓前請核查未確認的訓練要求：' },
  ['Could not record resume request: '] = { '无法记录续训请求：', '無法記錄續訓要求：' },
  ['Resume started, but its pending marker remains: '] = { '续训已启动，但待确认标记仍在：', '續訓已啟動，但待確認標記仍在：' },
  ['Resume outcome is not acknowledged. Use H to inspect request '] = { '续训结果尚未确认。请按 H 核查请求 ', '續訓結果尚未確認。請按 H 核查要求 ' },
  ['This CLI cannot pin an evaluation checkpoint.'] = { '当前 CLI 无法固定评估检查点。', '目前 CLI 無法固定評估檢查點。' },
  ['A verified checkpoint identity is required for evaluation.'] = { '评估需要可核对的检查点身份。', '評估需要可核對的檢查點身分。' },
  ['This CLI cannot pin an export checkpoint.'] = { '当前 CLI 无法固定导出检查点。', '目前 CLI 無法固定匯出檢查點。' },
  ['A verified checkpoint identity is required for export.'] = { '导出需要可核对的检查点身份。', '匯出需要可核對的檢查點身分。' },
  ['Export destination exists or cannot be checked: '] = { '导出目标已存在或无法检查：', '匯出目標已存在或無法檢查：' },
  ['Export outcome is not acknowledged. Inspect the destination before retrying.'] = { '导出结果尚未确认；重试前请检查目标目录。', '匯出結果尚未確認；重試前請檢查目標目錄。' },
  ['Source editing requires the saved project model, input, and a current configuration revision. Check project status first.'] = { '源码编辑需要已保存的模型、输入及当前配置修订；请先检查项目状态。', '原始碼編輯需要已儲存的模型、輸入及目前設定修訂；請先檢查專案狀態。' },
  ['Cannot verify source edit capabilities. No source was changed.'] = { '无法核对源码编辑能力；源码未修改。', '無法核對原始碼編輯能力；原始碼未修改。' },
  ['The CLI does not advertise '] = { 'CLI 未声明此能力，无法提交：', 'CLI 未宣告此能力，無法提交：' },
  ['Project changed; start a new preview.'] = { '项目已变化，请重新预览。', '專案已變更，請重新預覽。' },
  ['Candidate evidence is incomplete or stale; source cannot be committed. Review the report or edit manually.'] = { '候选证据不完整或已过期，不能提交源码；请查看报告或手动编辑。', '候選證據不完整或已過期，不能提交原始碼；請查看報告或手動編輯。' },
  ['Context changed; source was not changed. Preview again.'] = { '上下文已变化，源码未修改；请重新预览。', '上下文已變更，原始碼未修改；請重新預覽。' },
  ['Assembly requires the saved project model and representative input. Check project status first.'] = { '装配需要已保存的模型和代表输入；请先检查项目状态。', '裝配需要已儲存的模型和代表輸入；請先檢查專案狀態。' },
  ['Use object entries with a slot name and nonempty string value.'] = { '请使用接缝名和非空字符串值。', '請使用接縫名稱和非空字串值。' },
  ['Remove must be a list of slots.'] = { 'remove 必须是接缝列表。', 'remove 必須是接縫清單。' },
  ['Invalid slot.'] = { '接缝名称无效。', '接縫名稱無效。' },
  ['Parameters must be an object.'] = { '参数必须是对象。', '參數必須是物件。' },
  ['Component parameter names must be SLOT.NAME.'] = { '组件参数名必须为 SLOT.NAME。', '元件參數名稱必須為 SLOT.NAME。' },
  ['Invalid component parameter.'] = { '组件参数无效。', '元件參數無效。' },
  ['No assembly changes were provided.'] = { '没有提供装配修改。', '沒有提供裝配修改。' },
  ['This CLI does not support the requested guarded assembly change.'] = { '当前 CLI 不支持所请求的条件装配修改。', '目前 CLI 不支援所要求的條件裝配修改。' },
  ['Project changed; prepare assembly again.'] = { '项目已变化，请重新准备装配。', '專案已變更，請重新準備裝配。' },
  ['Assembly preview did not establish a current candidate; nothing saved.'] = { '装配预览未确认当前候选；未保存。', '裝配預覽未確認目前候選；未儲存。' },
  ['Assembly preview did not confirm the requested route.'] = { '装配预览未确认所请求的输入接线。', '裝配預覽未確認所要求的輸入接線。' },
  ['Assembly preview did not confirm the requested output mode.'] = { '装配预览未确认所请求的输出方式。', '裝配預覽未確認所要求的輸出方式。' },
  ['Assembly outcome requires inspection before retrying.'] = { '装配结果需要核查；请检查后再重试。', '裝配結果需要核查；請檢查後再重試。' },
  ['Invalid assembly JSON.'] = { '装配 JSON 无效。', '裝配 JSON 無效。' },
  ['Unknown assembly field: '] = { '未知装配字段：', '未知裝配欄位：' },
  ['Choose at least one parameter to change.'] = { '请至少选择一个要修改的参数。', '請至少選擇一個要修改的參數。' },
  ['Invalid parameter name: '] = { '参数名称无效：', '參數名稱無效：' },
  ['Cannot encode parameter: '] = { '无法编码参数：', '無法編碼參數：' },
  ['Run a current forward pass before marking an observed output.'] = { '标记输出前请先运行当前模型的前向。', '標記輸出前請先執行目前模型的前向。' },
  ['No project-source output ports were reported in this forward pass.'] = { '此次前向未报告项目源码输出端口。', '此次前向未回報專案原始碼輸出連接埠。' },
  ['The observation changed; run the forward pass again before marking.'] = { '观察结果已变化，请重新前向后再标记。', '觀察結果已變更，請重新前向後再標記。' },
  ['Wait for the current operation.'] = { '请等待当前操作完成。', '請等待目前操作完成。' },
  ['Rule parameters must be a JSON object.'] = { '规则参数必须是 JSON 对象。', '規則參數必須是 JSON 物件。' },
  ['Invalid rule parameter name.'] = { '规则参数名称无效。', '規則參數名稱無效。' },
  ['Invalid rule parameter value.'] = { '规则参数值无效。', '規則參數值無效。' },
  ['Range needs a minimum or maximum.'] = { '范围至少需要一个上界或下界。', '範圍至少需要一個上界或下界。' },
  ['Rule ID is ambiguous; refresh before editing.'] = { '规则 ID 不唯一；编辑前请刷新。', '規則 ID 不唯一；編輯前請重新整理。' },
  ['Cannot encode the selected rule.'] = { '无法编码所选规则。', '無法編碼所選規則。' },
  ['Rule removal was not acknowledged. Refresh rules before retrying.'] = { '规则移除结果尚未确认；重试前请刷新规则。', '規則移除結果尚未確認；重試前請重新整理規則。' },
  ['A saved project and current configuration revision are required. Check project status first.'] = { '需要已保存的项目及当前配置修订；请先检查项目状态。', '需要已儲存的專案及目前設定修訂；請先檢查專案狀態。' },
  ['Project changed; prepare the trial again.'] = { '项目已变化，请重新准备试跑。', '專案已變更，請重新準備試跑。' },
  ['This CLI cannot verify the project configuration before a trial.'] = { '当前 CLI 无法在试跑前核对项目配置。', '目前 CLI 無法在試跑前核對專案設定。' },
  ['Cannot read pending requests: '] = { '无法读取待确认请求：', '無法讀取待確認要求：' },
  ['Could not remove pending marker: '] = { '无法移除待确认标记：', '無法移除待確認標記：' },
  ['Training parameters must be a JSON object.'] = { '训练参数必须是 JSON 对象。', '訓練參數必須是 JSON 物件。' },
  ['Invalid training parameter name.'] = { '训练参数名称无效。', '訓練參數名稱無效。' },
  ['Invalid training parameter value.'] = { '训练参数值无效。', '訓練參數值無效。' },
  ['Project changed; prepare training again.'] = { '项目已变化，请重新准备训练。', '專案已變更，請重新準備訓練。' },
  ['This CLI cannot safely start and track this training request.'] = { '当前 CLI 无法安全启动并追踪此训练请求。', '目前 CLI 無法安全啟動並追蹤此訓練要求。' },
  ['Resolve pending training requests before starting: '] = { '启动前请核查未确认的训练请求：', '啟動前請核查未確認的訓練要求：' },
  ['Could not record the training request: '] = { '无法记录训练请求：', '無法記錄訓練要求：' },
  ['Training started, but its pending marker remains: '] = { '训练已启动，但待确认标记仍在：', '訓練已啟動，但待確認標記仍在：' },
  ['Training outcome is not acknowledged. Use H to inspect request '] = { '训练结果尚未确认。请按 H 核查请求 ', '訓練結果尚未確認。請按 H 核查要求 ' },
  ['This CLI cannot pin the supervision revision.'] = { '当前 CLI 无法固定监督修订。', '目前 CLI 無法固定監督修訂。' },
  ['Cannot verify supervision rules; training was not started.'] = { '无法核对监督规则；未启动训练。', '無法核對監督規則；未啟動訓練。' },
  ['Model-Maven: file is not readable: '] = { 'Model-Maven：文件不可读：', 'Model-Maven：檔案無法讀取：' },
  ['This project has a configuration or its state is unknown. No changes were made.'] = { '项目已有配置或状态未知；未作修改。', '專案已有設定或狀態未知；未作修改。' },
  ['Project setup preview is unavailable; no files were changed.'] = { '项目配置预览不可用；文件未修改。', '專案設定預覽無法使用；檔案未修改。' },
  ['Model-Maven: wait for the current operation.'] = { 'Model-Maven：请等待当前操作完成。', 'Model-Maven：請等待目前操作完成。' },
  ['This project has a configuration or its state is unknown; creation will not overwrite it.'] = { '项目已有配置或状态未知；创建不会覆盖它。', '專案已有設定或狀態未知；建立不會覆蓋它。' },
  ['Model-Maven: this CLI cannot safely create a model, or its capability report is invalid.'] = { 'Model-Maven：CLI 无法安全创建模型，或能力报告无效。', 'Model-Maven：CLI 無法安全建立模型，或能力報告無效。' },
  ['Model-Maven: creation plan unavailable; nothing was created.'] = { 'Model-Maven：创建计划不可用；未创建文件。', 'Model-Maven：建立計畫無法使用；未建立檔案。' },
  ['Wait for the operation to finish before switching projects.'] = { '请等待操作完成后再切换项目。', '請等待操作完成後再切換專案。' },
}

-- Prefer the most specific prefix and avoid rebuilding the search list for
-- each notification. Dynamic detail after a known prefix remains untouched.
local message_prefixes = {}
for prefix in pairs(messages) do
  if prefix:sub(-1) == ' ' or prefix:sub(-1) == '\n' then message_prefixes[#message_prefixes + 1] = prefix end
end
table.sort(message_prefixes, function(a, b)
  return #a == #b and a < b or #a > #b
end)

function M.message(value)
  if type(value) ~= 'string' then return value end
  local language = M.current()
  if language == 'en' then return value end
  local index = language == 'zh-tw' and 2 or 1
  local direct = messages[value]
  if direct then return direct[index] end
  for _, prefix in ipairs(message_prefixes) do
    if value:sub(1, #prefix) == prefix then return messages[prefix][index] .. value:sub(#prefix + 1) end
  end
  return value
end

-- Backend diagnostics may contain dynamic node names and dimensions. Keep the
-- original text intact; translate only a stable summary and suggested action.
local graph_summaries = {
    GRAPH_SHAPE_CONFLICT = {'Shape mismatch; inspect the affected ports.', '形状不匹配，请检查相关端口。', '形狀不相符，請檢查相關連接埠。'},
    GRAPH_DTYPE_ORDER_CONFLICT = {'Dtype source order conflicts; review the affected sequence.', '类型来源顺序冲突，请检查相关序列。', '型別來源順序衝突，請檢查相關序列。'},
    GRAPH_AXIS_INVALID = {'Axis is incompatible with the declared shape.', '轴与声明的形状不兼容。', '軸與宣告的形狀不相容。'},
    GRAPH_CYCLE = {'The graph contains a cycle; inspect its connections.', '图中存在环，请检查连接。', '圖中存在迴圈，請檢查連線。'},
    GRAPH_EDGE_INVALID = {'A connection is invalid; inspect its endpoints.', '连接无效，请检查两端。', '連線無效，請檢查兩端。'},
    GRAPH_INCOMPLETE = {'The graph is incomplete; inspect missing connections.', '图尚未完成，请检查缺失的连接。', '圖尚未完成，請檢查缺少的連線。'},
    GRAPH_REORDER_UNSUPPORTED = {'This sequence cannot be reordered here; edit the connections.', '此处无法重排该序列，请编辑连接。', '此處無法重排此序列，請編輯連線。'},
    GRAPH_BASE_CONFLICT = {'The saved graph changed; compare it before saving.', '已保存的图发生变化，请比较后再保存。', '已儲存的圖發生變更，請比較後再儲存。'},
    GRAPH_SOURCE_CONFLICT = {'The source changed; review it before continuing.', '源码发生变化，请检查后再继续。', '原始碼發生變更，請檢查後再繼續。'},
    GRAPH_DELIVERY_PARTIAL = {'Delivery is incomplete; inspect its status and resume or archive it.', '交付尚未完成，请检查状态后恢复或归档。', '交付尚未完成，請檢查狀態後復原或封存。'},
    GRAPH_AXIS_REQUIRED = {'Set the required axis for this operation.', '请为此操作设置所需的轴。', '請為此操作設定所需的軸。'},
    GRAPH_COMPONENT_INVALID = {'The component declaration is invalid; review its factory and parameters.', '组件声明无效，请检查工厂和参数。', '元件宣告無效，請檢查工廠和參數。'},
    GRAPH_DEFINITION_INVALID = {'The component definition is invalid; inspect its details.', '组件定义无效，请检查详情。', '元件定義無效，請檢查詳情。'},
    GRAPH_NODE_INVALID = {'The node declaration is invalid; inspect its details.', '节点声明无效，请检查详情。', '節點宣告無效，請檢查詳情。'},
    GRAPH_GROUP_INVALID = {'The display group is invalid; inspect its members and parent.', '显示分组无效，请检查成员和父分组。', '顯示分組無效，請檢查成員和父分組。'},
    GRAPH_INPUT_CONFLICT = {'The graph input conflicts with another declaration.', '图输入与其他声明冲突。', '圖輸入與其他宣告衝突。'},
    GRAPH_OUTPUT_INVALID = {'The graph output declaration is invalid.', '图输出声明无效。', '圖輸出宣告無效。'},
    GRAPH_OUTPUT_REQUIRED = {'Add the required graph output.', '请添加所需的图输出。', '請新增所需的圖輸出。'},
    GRAPH_DATA_INVALID = {'The data declaration is invalid; review the source and binding.', '数据声明无效，请检查来源和绑定。', '資料宣告無效，請檢查來源和繫結。'},
    GRAPH_DATA_REQUIRED = {'Choose a data source before continuing.', '继续前请选择数据来源。', '繼續前請選擇資料來源。'},
    GRAPH_OBJECTIVE_INVALID = {'The loss or target declaration is invalid.', '损失或目标声明无效。', '損失或目標宣告無效。'},
    GRAPH_TRAINING_INVALID = {'The training declaration is invalid; review its settings.', '训练声明无效，请检查设置。', '訓練宣告無效，請檢查設定。'},
    GRAPH_SUPERVISION_INVALID = {'The supervision reference is invalid; inspect the referenced file.', '监督引用无效，请检查引用文件。', '監督參照無效，請檢查參照檔案。'},
    GRAPH_EDIT_INVALID = {'The proposed graph edit is invalid; no change was applied.', '拟进行的图编辑无效，未应用修改。', '擬進行的圖編輯無效，未套用修改。'},
    GRAPH_OPERATIONS_INVALID = {'The graph edit operations are invalid.', '图编辑操作无效。', '圖編輯操作無效。'},
    GRAPH_OPERATIONS_CONFLICT = {'The edit request changed; plan it again.', '编辑请求已变化，请重新规划。', '編輯要求已變更，請重新規劃。'},
    GRAPH_PLAN_CONFLICT = {'The plan is stale; review a new plan.', '方案已过期，请重新审核。', '方案已過期，請重新檢視。'},
    GRAPH_CONFLICT = {'The graph changed; compare it before continuing.', '图已变化，请比较后再继续。', '圖已變更，請比較後再繼續。'},
    GRAPH_BIND_CONFLICT = {'The project binding changed; review it again.', '项目绑定已变化，请重新审核。', '專案繫結已變更，請重新檢視。'},
    GRAPH_DELIVERY_CONFLICT = {'Delivery encountered a conflict; inspect its status.', '交付遇到冲突，请检查状态。', '交付遇到衝突，請檢查狀態。'},
    GRAPH_DELIVERY_PENDING = {'A previous delivery is pending; inspect it before starting another.', '先前交付尚未完成，请先检查。', '先前交付尚未完成，請先檢查。'},
    GRAPH_DELIVERY_MISSING = {'The requested delivery record is unavailable.', '请求的交付记录不可用。', '要求的交付紀錄無法使用。'},
    GRAPH_READ_FAILED = {'The graph file could not be read.', '无法读取图文件。', '無法讀取圖檔。'},
    GRAPH_WRITE_FAILED = {'The graph could not be saved; inspect the original error.', '无法保存图，请检查原始错误。', '無法儲存圖，請檢查原始錯誤。'},
    GRAPH_SOURCE_WRITE_FAILED = {'The source could not be written; inspect the original error.', '无法写入源码，请检查原始错误。', '無法寫入原始碼，請檢查原始錯誤。'},
}
function M.graph_diagnostic(diagnostic, fallback)
  if type(diagnostic) ~= 'table' then return fallback end
  local summary = graph_summaries[diagnostic.code]
  local detail = type(diagnostic.message) == 'string' and diagnostic.message or nil
  if not summary then return detail or fallback end
  local translated = M.text(summary[1], summary[2], summary[3])
  if detail and detail ~= '' then return translated .. ' ' .. detail end
  return translated
end

function M.choose(on_change)
  local choices = { labels.system, labels.en, labels['zh-cn'], labels['zh-tw'] }
  local keys = { 'system', 'en', 'zh-cn', 'zh-tw' }
  vim.ui.select(choices, { prompt = M.text('Language', '语言', '語言') }, function(choice, index)
    if not choice or not index then return end
    local language = keys[index]
    selected = language
    local document, path, original = read_document()
    local saved, message = false, nil
    if document and path then
      local parent = vim.fn.fnamemodify(path, ':h')
      if vim.fn.mkdir(parent, 'p', 448) == 1 or vim.fn.isdirectory(parent) == 1 then
        document.language = language
        local encoded = vim.json.encode(document) .. '\n'
        local temporary = path .. '.tmp-' .. tostring(vim.fn.getpid()) .. '-' .. tostring(uv.hrtime())
        local fd, err = uv.fs_open(temporary, 'wx', 384)
        if fd then
          local written, write_err = uv.fs_write(fd, encoded, 0)
          uv.fs_close(fd)
          if written == #encoded then
            local current, current_path, current_raw = read_document()
            if current and current_path == path and current_raw == original then
              local renamed, rename_err = uv.fs_rename(temporary, path)
              if renamed then saved = true else message = rename_err end
            else
              message = M.text('preferences changed while saving; retry', '保存期间偏好已变化，请重试', '儲存期間偏好已變更，請重試')
            end
          else message = write_err or M.text('incomplete write', '写入不完整', '寫入不完整') end
          if not saved then uv.fs_unlink(temporary) end
        else message = err end
      else message = M.text('cannot create preference directory', '无法创建偏好设置目录', '無法建立偏好設定目錄') end
    else message = M.text('preferences unavailable or invalid', '偏好设置不可用或格式无效', '偏好設定無法使用或格式無效') end
    if not saved then
      vim.notify(M.text('Language changed for this session only: ', '语言仅在本次会话生效：', '語言僅在本次工作階段生效：') .. tostring(message), vim.log.levels.WARN)
    end
    if on_change then on_change() end
  end)
end

return M
