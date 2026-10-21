# 修复源码安装时 catlass 缺失

已根据服务器日志定位：`dependency catlass is missing` 后，构建脚本执行 `git submodule update`；传输后的 `/workspace/dyntralb` 没有可用 Git 仓库信息，因此 `fetch failed`，导致 editable wheel 构建失败。

本分支现已在构建脚本中接入离线恢复：缺少 CATLASS 时，先校验仓库随附压缩包的 SHA256，再自动解压。通过 Git 克隆或下载完整分支源码后，在原服务容器或虚拟环境执行：

```bash
cd /workspace/dyntralb
python -m pip install -e . --no-deps --no-build-isolation
```

请保留源码根目录的 `.gitmodules`、`catlass-41bf90da-headers.tar.gz` 和 `catlass-41bf90da-headers.tar.gz.sha256`。离线包缺失时，有 Git 元数据的 checkout 会回退到原来的子模块下载；没有 Git 元数据则会报出需要补齐的文件。压缩包校验失败时停止构建，不会静默切换依赖版本。不会修改用户的全局 Git 配置。

若服务器还在使用旧构建脚本，只需将 `catlass-41bf90da-headers.tar.gz` 上传到服务器 `/workspace/dyntralb/`，手动解压后再安装：

```bash
cd /workspace/dyntralb
tar -xzf catlass-41bf90da-headers.tar.gz
ls csrc/third_party/catlass/include/catlass/catlass.hpp
python -m pip install -e . --no-deps --no-build-isolation
```

包内路径已经包含 `csrc/third_party/catlass/`，请直接在源码根目录解压，不要额外使用 `--strip-components`。`ls` 应输出该头文件路径；若失败，先确认解压位置再安装。保持原来的 CANN、SOC_VERSION、vLLM 和 TorchNPU 设置。此操作不要求重新初始化整个源码目录的 Git，也不需要设置 `GIT_DISCOVERY_ACROSS_FILESYSTEM`。

## 版本和范围

- 来源：[CATLASS 官方仓库](https://gitcode.com/cann/catlass.git)
- 固定提交： `41bf90da655bba3c66d0acd7e00abe33960ecfd6`，与此版 `.gitmodules` 和 submodule gitlink 一致。
- 包含全部 233 个 `include/` 文件、上游 LICENSE、README 和许可通知。
- 仅补全 CATLASS 构建头文件；不包含 CANN、编译器或已编译算子。CATLASS 自身单元测试的 googletest 子模块不包含在内，当前 vllm-ascend 构建不需要它。
- 已验证解压后所有头文件与固定提交逐个一致，并在无 `.git` 的临时目录测试自动恢复、重复运行、校验失败、损坏的压缩包和依赖缺失报错；另用本地 Git 仓库测试子模块下载回退，不依赖服务器联网。
- 未在 Ascend 服务器执行完整构建，后续其他构建错误需要按新的错误处理。

`listing git files failed` 和 `0.0.0` 可能仍出现在无 Git 信息的源码构建日志中；当前附件里的致命退出来自 catlass 的 submodule fetch，补头文件解决的是这一处退出。

## makeself 及其他 CMake 构建依赖

若后续出现 `makeself-populate-download`、`Build step for makeself failed`，说明失败在 makeself 下载或验证步骤。后面的 `No rule to make target prepare_build` 是配置未完成导致的连带错误。

本分支现将 makeself、JSON、Abseil 和 Protobuf 的固定版本源码包保存在 [构建依赖目录](csrc/cmake/third_party/archives/README.md)，CMake 会先校验并使用这些本地文件。请更新完整分支源码，保留 `csrc/cmake/third_party/archives/` 及 `SHA256SUMS`，然后在原服务环境中重新运行前面的安装命令。已有的 CANN、芯片型号和模型启动配置继续沿用。

同时修正了 makeself 的执行权限路径、JSON 本地压缩包的实际位置，以及配置失败后仍继续执行 `make` 的行为。无需因为这段错误将 CMake 降级，也无需删除原服务的已编译算子来重试。完整 Ascend 编译尚待服务器验证。
