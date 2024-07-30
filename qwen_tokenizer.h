#pragma once

#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace qwen {

struct QwenTokenizerConfig {
  // for tokenizer
  int eos_token_id;
  int pad_token_id;
  int im_start_id;
  int im_end_id;
};

class IQwenTokenizer {
  public:

    virtual auto encode(const std::string& text, int max_length) const->std::vector<int> = 0;

    virtual auto decode(const std::vector<int> &ids) const -> std::string  = 0;

    virtual auto encode_history(const std::vector<std::string> &history, int max_length) const -> std::vector<int> = 0;

    virtual auto build_prompt(const std::vector<std::string> &history) const -> std::string = 0;

    virtual auto is_special_id(int id) const -> bool = 0;

    static std::shared_ptr<IQwenTokenizer> make(const std::string& tiktoken_path, const QwenTokenizerConfig& config);
};


} // namespace qwen
