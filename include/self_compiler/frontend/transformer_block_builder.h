#pragma once

#include "self_compiler/ir/graph.h"

namespace self_compiler::frontend {

struct TransformerBlockSpec {
    int batch = 1;
    int sequence_length = 16;
    int hidden_size = 2048;
    int intermediate_size = 8192;
    int num_attention_heads = 32;
    int num_key_value_heads = 8;
    int vocab_size = 128256;
};

class TransformerBlockBuilder {
public:
    ir::Graph Build(const TransformerBlockSpec& spec) const;
};

}  // namespace self_compiler::frontend
