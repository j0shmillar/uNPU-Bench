#include "model.h"
#include "k230_math.h"

// for image
Model::Model(const char *kmodel_file) : AIBase(kmodel_file, "Yolo")
{
    model_name_ = "Yolo";
    ai2d_out_tensor_ = get_input_tensor(0);
}

void Model::inference(unsigned char* random_img, int width, int height)
{
    // Convert random_img to a vector of unsigned chars (BGR format)
    std::vector<unsigned char> img_vec;
    img_vec.reserve(width * height * 3);  // Assuming 3 channels (BGR)

    // Fill the vector with the image data in BGR format
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            img_vec.push_back(random_img[(i * width + j) * 3 + 0]);  // B
            img_vec.push_back(random_img[(i * width + j) * 3 + 1]);  // G
            img_vec.push_back(random_img[(i * width + j) * 3 + 2]);  // R
        }
    }

    // Convert the vector to a cv::Mat (BGR format)
    cv::Mat img_mat(height, width, CV_8UC3, img_vec.data());

    // Convert BGR to RGB and HWC to CHW format directly without resizing
    std::vector<uint8_t> chw_vec;
    Utils::bgr2rgb_and_hwc2chw(img_mat, chw_vec);  // Convert the image to CHW format

    // Create input tensor (matching your example)
    dims_t in_shape{1, 3, height, width};  // Assuming 3 channels for RGB
    auto ai2d_in_tensor = host_runtime_tensor::create(typecode_t::dt_uint8, in_shape, hrt::pool_shared).expect("cannot create input tensor");

    // Copy data from the CHW vector into the tensor's buffer
    auto input_buf = ai2d_in_tensor.impl()->to_host().unwrap()->buffer().as_host().unwrap().map(map_access_::map_write).unwrap().buffer();
    memcpy(reinterpret_cast<char *>(input_buf.data()), chw_vec.data(), chw_vec.size());

    // Synchronize the tensor (write back)
    hrt::sync(ai2d_in_tensor, sync_op_t::sync_write_back, true).expect("write back input failed");

    // Now the tensor is ready for inference
    ai2d_out_tensor_ = ai2d_in_tensor;  // Assuming the output tensor is the same for now

    // Run inference
    this->run();
    this->get_output();
}

Model::~Model()
{
}

