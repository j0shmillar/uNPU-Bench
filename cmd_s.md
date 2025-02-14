## model
### vela

```
vela --accelerator-config ethos-u55-64 --recursion-limit 2000 --optimise {Size|Performance} ./model/{model}/quant/{model}_full_integer_quant.tflite --output-dir ./model/{model}/vela_{size|perf} 
```

### eiq

```
/opt/nxp/eIQ_Toolkit_v1.13.1/bin/neutron-converter/MCU_SDK_2.16.000/neutron-converter --input {model}_full_integer_quant.tflite --output {model}_full_integer_quant_eiq.tflite
```

### mlir

```
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
model_transform.py --model_name {model} --model_def model/{model}/{model}.onnx --mlir model/{model}/{model}.mlir
```

## npu
### luckfox
on linux machine, do:
```
cd npu/luckfox/{model}
./build.sh
```
then
```
adb push luckfox_pico_{model}_demo root
adb shell
cd root/luckfox_pico_{model}_demo
./luckfox_pico_{model} model/{model}.rknn
```
