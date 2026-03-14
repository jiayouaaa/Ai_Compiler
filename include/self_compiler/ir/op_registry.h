#pragma once

#include <map>
#include <string>
#include <vector>

#include "self_compiler/ir/operation.h"

namespace self_compiler::ir {

// 描述一个 op 需要的一个 weight 输入（如卷积核、偏置）
struct WeightDesc {
    std::string name_suffix;   // 名字后缀（如 "weight", "bias", "wq"）
    bool required = true;      // true = 必须有，false = 可选（如 bias）
};

// 描述一个算子的静态属性
// 所有 pass 通过查询 OpRegistry 获取这些信息，而不是各自写 switch
//
// op.inputs 的排列约定：activation 在前，weight 在后
//   例：Conv2D.inputs = [activation, weight, bias]
//                        ├ activation ┤├── weight ──┤
struct OpInfo {
    std::string name;                              // 算子显示名（如 "Conv2D"）
    int num_activation_inputs = 0;                 // activation（数据流）输入数
    int num_outputs = 0;                           // 输出数（-1 表示不限制）
    std::vector<WeightDesc> weight_inputs;         // weight（参数流）输入描述
    std::vector<std::string> required_attributes;  // 必须存在的 attribute 列表
    bool npu_supported = false;                    // Ethos-U55 是否支持

    // 从 activation 数量 + weight 描述自动计算 inputs 的合法范围
    int MinInputs() const {
        int required_weights = 0;
        for (const auto& w : weight_inputs) {
            if (w.required) ++required_weights;
        }
        return num_activation_inputs + required_weights;
    }
    int MaxInputs() const {
        return num_activation_inputs + static_cast<int>(weight_inputs.size());
    }
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
