from sklearn.covariance import EllipticEnvelope
import numpy as np
aaa = np.array([[1,2,-10000,3,4,6,7,8,90,100,5000],
                [1000,2000,3,4000,5000,6000,7000,8,9000,10000,1001]])
aaa=np.transpose(aaa)
print(aaa.shape)

# outlier = EllipticEnvelope(contamination=.2) #20%의 오염도를 찾아라
# outlier = EllipticEnvelope(contamination=.1) #10%의 오염도를 찾아라
outlier = EllipticEnvelope(contamination=.3) #30%의 오염도를 찾아라
outlier.fit(aaa)

print(outlier.predict(aaa)) # 오염된게 -1 = outlier
