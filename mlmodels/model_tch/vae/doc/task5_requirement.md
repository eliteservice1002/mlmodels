#### 5th submission   #####################################################
A new model is in  :   models/Beta_VAE_fft/
We change the input X as follow :

   Previously : X is 2D tensor (xmax, ymax)  with 0/1 values
     (32,32) ---> (256,1)   --> (  32,32)



######## New  :      #####################################################
      Encoder Input :
          X_2D image

      Encoder output :
          X_2D :    [a,w,b,c] * 2  (mean, variance)
           dims:    (4,1) * 2


      Decoder Input :
          X_2D :  [a,w,b,c] * 2  (mean, variance)

          ??Function[a,w,b,c]  -->   a.cos(w.t + b) + c    in 2D Tensor.
          Having Differentiable Function to compute cosinus....


      Decoder output :
          X_2D : Image


######## Goal is to create a diffenerational tensor generation of cosinus :

     ##### Middle generator data :
         Function[a,w,b,c]   -->      X_2D (32, 32) = a.cost(w.t + b)

    correction: input is (64,64)

















