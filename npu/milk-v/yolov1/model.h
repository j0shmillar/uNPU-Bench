#ifndef _MODEL_H
#define _MODEL_H

#include <iostream>
#include <vector>

#include "utils.h"
#include "ai_base.h"

using std::vector;

/**
 * @brief 人脸检测
 * 主要封装了对于每一帧图片，从预处理、运行到后处理给出结果的过程
 */
class Model : public AIBase
{
public:
    /**
     * @param kmodel_file kmodel文件路径
     */
    Model(const char *kmodel_file);

    /**
     * @return None
     */
    ~Model();

    /**
     * @brief kmodel推理
     * @return None
     */
    void inference(unsigned char*, int, int);

private:
    std::unique_ptr<ai2d_builder> ai2d_builder_; // ai2d构建器
    runtime_tensor ai2d_in_tensor_;              // ai2d输入tensor
    runtime_tensor ai2d_out_tensor_;             // ai2d输出tensor
    uintptr_t vaddr_;                            // isp的虚拟地址
    FrameCHWSize isp_shape_;                     // isp对应的地址大小
};

#endif