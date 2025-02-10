## Vela

```
vela --accelerator-config ethos-u55-64 --recursion-limit 2000 --optimise {Size|Performance} ./model/{model}/quant/{model}_full_integer_quant.tflite --output-dir ./model/{model}/vela_{size|perf} 
```
