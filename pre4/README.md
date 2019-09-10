# 洗濯表示を識別する（4種類）

## 目的
- 液温は40℃を限度とし、洗濯機で弱い洗濯ができる

![1](https://user-images.githubusercontent.com/20394831/64636916-ab16c300-d43d-11e9-9843-fbb742f2983c.png)

- タンブル乾燥禁止

![1](https://user-images.githubusercontent.com/20394831/64636937-b36efe00-d43d-11e9-872d-aaf88210fc77.png)

- 塩素系及び酸素系漂白剤の使用禁止

![1](https://user-images.githubusercontent.com/20394831/64636947-b8cc4880-d43d-11e9-9ef9-c8ba52517194.png)

- 底面温度150℃を限度としてアイロン仕上げができる

![1](https://user-images.githubusercontent.com/20394831/64636956-bec22980-d43d-11e9-8ee3-e0fe271020a6.png)


上、4種類の洗濯表示を画像認識により識別する

## 方法
1. 学習用の画像を集める
2. それぞれ水増しする
3. DeepLearningする
4. 試してみる

### 学習用の画像
- 液温は40℃を限度とし、洗濯機で弱い洗濯ができる

![1](https://user-images.githubusercontent.com/20394831/64637069-fd57e400-d43d-11e9-8d5e-5788530e95ea.png)
![3_usui_1](https://user-images.githubusercontent.com/20394831/64637087-0943a600-d43e-11e9-8037-7a70630861aa.png)
![4_usui_1](https://user-images.githubusercontent.com/20394831/64637101-12347780-d43e-11e9-9523-6310528db336.png)

- タンブル乾燥禁止
- 塩素系及び酸素系漂白剤の使用禁止
- 底面温度150℃を限度としてアイロン仕上げができる

### それぞれ水増しする
- 明るさ
- ノイズ
- 回転
- 並進移動

40_weak  
3枚 -> 5716枚

![Screenshot from 2019-09-11 00-25-07](https://user-images.githubusercontent.com/20394831/64637472-d4841e80-d43e-11e9-8b69-eecfea7b424a.png)

4種類合計15,716枚に

### DeepLearningする
画像データの8割を訓練用、2割をテスト用に分ける。  

訓練データ、テストデータ共に100%近くの精度に  

[0.0022653688505458467, 0.9996818326439707]  

### 試してみる
学習に関係ない画像で試す  

["40_weak","donot_tumble_dry","ironing_upto150","not_bleachable"]  

40_weak

![40_weak_1](https://user-images.githubusercontent.com/20394831/64637885-b2d76700-d43f-11e9-8d26-c4b8b793a93d.jpg)
![image](https://user-images.githubusercontent.com/20394831/64638067-0cd82c80-d440-11e9-9ba2-6fb54a10755d.png)


![40_weak_2](https://user-images.githubusercontent.com/20394831/64638175-43ae4280-d440-11e9-859b-7bc7ea2f0c0c.jpg)
![image](https://user-images.githubusercontent.com/20394831/64638172-40b35200-d440-11e9-8297-90b5037440bc.png)

![40_weak_3](https://user-images.githubusercontent.com/20394831/64638362-8708b100-d440-11e9-9901-358974f87800.jpg)
![image](https://user-images.githubusercontent.com/20394831/64638366-88d27480-d440-11e9-9c14-e9613054b350.png)

3枚中3枚認識  

donot_tumble_dry  

![donot_tumble_dry_1](https://user-images.githubusercontent.com/20394831/64638463-c8995c00-d440-11e9-83e2-9e01940075ae.jpg)
![image](https://user-images.githubusercontent.com/20394831/64638509-dd75ef80-d440-11e9-8a7a-c56a780e0f84.png)

![donot_tumble_dry_2](https://user-images.githubusercontent.com/20394831/64638539-e961b180-d440-11e9-8b06-5a9e1ef0e345.jpg)
![image](https://user-images.githubusercontent.com/20394831/64638562-f9799100-d440-11e9-8008-ecf82a96b4c7.png)

![donot_tumble_dry_3](https://user-images.githubusercontent.com/20394831/64638580-026a6280-d441-11e9-97ae-a43fb940f7f9.jpg)
![image](https://user-images.githubusercontent.com/20394831/64638614-101fe800-d441-11e9-9f1b-d4abc955f97d.png)

3枚中1枚認識  

ironing_upto150  

![ironing_upto150_1](https://user-images.githubusercontent.com/20394831/64638666-2ded4d00-d441-11e9-9eb3-e51527f60bf3.jpg)
![image](https://user-images.githubusercontent.com/20394831/64638697-4198b380-d441-11e9-81f7-28db6d6fbe76.png)

![ironing_upto150_2](https://user-images.githubusercontent.com/20394831/64638716-4a898500-d441-11e9-804e-148ad3d14402.jpg)
![image](https://user-images.githubusercontent.com/20394831/64638747-58d7a100-d441-11e9-9f0c-65a1e25b6d3b.png)

![ironing_upto150_3](https://user-images.githubusercontent.com/20394831/64638762-63923600-d441-11e9-9ab5-3da7790e91d0.jpg)
![image](https://user-images.githubusercontent.com/20394831/64638798-7b69ba00-d441-11e9-9109-db065022c487.png)

3枚中3枚認識  

not_bleachable  

![not_bleachable_1](https://user-images.githubusercontent.com/20394831/64638861-9b00e280-d441-11e9-9cd6-a0133d8ad6d8.jpg)
![image](https://user-images.githubusercontent.com/20394831/64638887-a94efe80-d441-11e9-9206-6fdbcd3ffea0.png)

![not_bleachable_2](https://user-images.githubusercontent.com/20394831/64638932-bec42880-d441-11e9-9dc8-fc518c098c22.jpg)
![image](https://user-images.githubusercontent.com/20394831/64638976-d69bac80-d441-11e9-9051-a80e220c788d.png)

2枚中2枚認識
