# haihua_2021
海华AI挑战赛阅读理解，pytorch+gpt2+层次attention，用欧式距离作四分类。

用滑窗机制将超长文本划分为子段，两级attention：段内 word level + 段间 slice level。

段长取512时总计算量非常大，请保证你的GPU有 24G 以上的显存，显存遭不住的话可以使用更小的段长。

GPT2来自杜老师的GPT2-chinese开源项目：https://github.com/Morizeyao/GPT2-Chinese
