from pyspark.sql.functions import col, when, min, max
from pyspark.sql.types import DoubleType
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler,MinMaxScaler
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import array
from pyspark.sql.functions import concat
from pyspark.ml.stat import Correlation

spark = SparkSession.builder.appName("Final Code").getOrCreate()
df = spark.read.options(inferSchema='True',delimiter=',',header='True').csv("zohre:/ML_hw_mydataset_numeric_positive.csv")




#STEP 1:Select top 15 using correlation:
assembler = VectorAssembler(inputCols=df.columns[:-1], outputCol="features")
df = assembler.transform(df)
# Calculate correlation:
correlation_matrix = Correlation.corr(df, "features")
corr_matrix = correlation_matrix.select("pearson(features)").collect()[0][0]
corr_array = corr_matrix.toArray()
#top 15:
label_column = len(df.columns) - 3
corr_values = [abs(corr_array[i][label_column]) for i in range(len(corr_array) - 1)]
sorted_corr = sorted(zip(corr_values, df.columns[:-1]), reverse=True)[:15]
selected_features = [col(column) for _, column in sorted_corr]

selected_data = df.select(selected_features + [col("y")])
selected_data.show()


#Step 2:PCA
cols_to_pca = selected_data.columns[:-1]
va = VectorAssembler(inputCols=cols_to_pca,outputCol="SS_features")
new_df=va.transform(selected_data)
pca = PCA(k=9, inputCol="SS_features", outputCol="pca_features")
model = pca.fit(new_df)
transformed = model.transform(new_df)
transformed = transformed.drop(*cols_to_pca)
transformed = transformed.drop("SS_features")
transformed.show(truncate=False)


#Step 3:Normalizion
mm_s=MinMaxScaler(inputCol="pca_features",outputCol="MinMax_SS")
Normalized_DF=mm_s.fit(transformed).transform(transformed)
Normalized_DF.select("MinMax_SS").show(truncate=False)


train, test = Normalized_DF.randomSplit([0.7, 0.3], seed=42)

#LR Model Making
lr = LogisticRegression(featuresCol='MinMax_SS', labelCol='y')
model = lr.fit(train)
myoutputs = model.evaluate(test)
predictions = model.transform(test)
print('LR - Calculated Accuracy:',myoutputs.accuracy)
print('LR - Calculated Precision:',myoutputs.precisionByLabel[1])
print('LR - Calculated Recall:',myoutputs.recallByLabel[1])
evaluator = BinaryClassificationEvaluator(labelCol='y')
print('LR - Test Area Under ROC (Accuracy based on BinaryClassification):', evaluator.evaluate(predictions))
predictions.groupBy("y","prediction").count().show()

#SVM Model Making:
from pyspark.ml.classification import LinearSVC

vector_assembler = VectorAssembler(inputCols=["MinMax_SS"], outputCol="features")
transformed_df = vector_assembler.transform(Normalized_DF)
training_data, test_data = transformed_df.randomSplit([0.7, 0.3], seed=42)
svm = LinearSVC(maxIter=40, regParam=0.1, labelCol='y')
svm_model = svm.fit(training_data)
myoutputs = svm_model.evaluate(test_data)
predictions = svm_model.transform(test_data)
print('SVM - Calculated Accuracy:',myoutputs.accuracy)
print('SVM - Calculated Precision:',myoutputs.precisionByLabel[1])
print('SVM - Calculated Recall:',myoutputs.recallByLabel[1])
evaluator = BinaryClassificationEvaluator(labelCol="y")
accuracy = evaluator.evaluate(predictions)
print("SVM - Test Area Under ROC (Accuracy based on BinaryClassification): {}".format(accuracy))

predictions.groupBy("y","prediction").count().show()