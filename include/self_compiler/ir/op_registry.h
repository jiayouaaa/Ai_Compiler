#pragma once

#include <map>
#include <string>
#include <vector>

#include "self_compiler/ir/operation.h"

namespace self_compiler::ir {

// 描述一个算子的静态属性（arity、必需 attribute、硬件支持等）
// 所有 pass 通过查询 OpRegistry 获取这些信息，而不是各自写 switch
struct OpInfo {
    std::string name;                              // 算子显示名（如 "Conv2D"）
    int min_inputs = 0;                            // 最少输入数
    int max_inputs = 0;                            // 最多输入数
    int num_outputs = 0;                           // 输出数（-1 表示不限制）
    std::vector<std::string> required_attributes;  // 必须存在的 attribute 列表
    bool npu_supported = false;                    // Ethos-U55 是否支持
};

// 算子注册表：OpKind → OpInfo 的全局映射
// 使用方式：
//   const auto* info = OpRegistry::Instance().Find(OpKind::kConv2D);
//   if (info) { /* 查到了 */ }
class OpRegistry {
public:
    // 获取全局单例（内部在首次调用时自动注册所有算子）
    static const OpRegistry& Instance();

    // 注册一个算子（只在初始化时调用）
    void Register(OpKind kind, OpInfo info);

    // 查询一个算子的信息，未注册返回 nullptr
    const OpInfo* Find(OpKind kind) const;

    // 获取所有已注册的算子
    const std::map<OpKind, OpInfo>& All() const { return registry_; }

private:
    OpRegistry() = default;
    std::map<OpKind, OpInfo> registry_;
};

}  // namespace self_compiler::ir
