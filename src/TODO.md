* reference.mat他を使用して、事前学習してみる。1.77 -> 1.56
* DepthwiseConv -> MLPじゃなくて、DepthwiseConv -> Convにしてみる。
* モデルを深くしてみる。
* カーネル・サイズを大きくしてみる。
* trim_meanの割合を変更してみる。

----

* number_of_modelsを30のまま、DepthwiseConvもConvも4回に戻す。
* 事後学習で低層のパラメーターを固定してみる。

----

* Normalizationを変更してみる。
* Poolingを変更してみる。
* DepthwiseConvを減らしてConvを増やしてみる。
* ノイズをやめてみる。
* 前段のDepthwiseConvをやめてみる。
* ドロップアウトとノイズを大きくしてみる。
* reference.matを使用して、行きと戻りの関係を調べてみる。
* カーネル・サイズをさらに大きくしてみる。
