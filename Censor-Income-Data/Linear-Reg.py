import pandas as pd
import tensorflow as tf
import tempfile as tmp
import urllib.request as url


train_file = tmp.NamedTemporaryFile()
test_file = tmp.NamedTemporaryFile()
url.urlretrieve("http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)
url.urlretrieve("http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)

CSV_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "gender",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income_bracket"]

df_train = pd.read_csv(train_file.name, names = CSV_COLUMNS, skipinitialspace= True)
df_test = pd.read_csv(test_file.name, names = CSV_COLUMNS, skipinitialspace= True, skiprows=1)


train_labels = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
test_labels = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

print("printing dtypes of train")
print(df_train.dtypes)


print("printing dtypes of test")
print(df_test.dtypes)

"""Input builder function"""
def input_fn(data_file, num_epochs, shuffle):

    df_data = pd.read_csv(data_file, names=CSV_COLUMNS, skipinitialspace= True)
    df_data = df_data.dropna(how='any', axis=0)

    labels = df_data['income_bracket'].apply(lambda x: ">50k" in x).astype(int)

    return tf.estimator.inputs.pandas_input_fn(x=df_data, y=labels, batch_size=100,
                                               num_epochs=num_epochs,shuffle=shuffle, num_threads=5)


"""Creating categorical columns"""
occupation = tf.feature_column.categorical_column_with_vocabulary_list(
    "occupation", [
        "Adm-clerical","Exec-managerial","Handlers-cleaners","Tech-support"
        "Prof-specialty","Other-service","Craft-repair"
        "Transport-moving","Farming-fishing","Machine-op-inspct","Sales","Protective-serv"
    ]
)
marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
    "marital_status", [
        "Married-civ-spouse", "Divorced", "Married-spouse-absent",
        "Never-married", "Separated", "Married-AF-spouse", "Widowed"
    ])

gender = tf.feature_column.categorical_column_with_vocabulary_list(
    "gender", ["Female", "Male"])

relationship = tf.feature_column.categorical_column_with_vocabulary_list(
    "relationship", [
        "Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
        "Other-relative"
    ])

workclass = tf.feature_column.categorical_column_with_vocabulary_list(
    "workclass", [
        "Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
        "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked"
    ])

education = tf.feature_column.categorical_column_with_vocabulary_list("education",
        ["Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th","Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th","Preschool", "12th"])

native_country = tf.feature_column.categorical_column_with_hash_bucket(
    "native_country", hash_bucket_size=1000)

""" Created Numeric columns"""
age = tf.feature_column.numeric_column('age')
education_num = tf.feature_column.numeric_column('education_num')
capital_gain = tf.feature_column.numeric_column('capital_gain')
capital_loss = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')

""" Creating buckets and crossed columns """
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
education_x_occupation = tf.feature_column.crossed_column(["education", "occupation"], hash_bucket_size=1000)
age_x_education_x_occupation = tf.feature_column.crossed_column([age_buckets, "education","occupation"],
                                                                hash_bucket_size=1000)

""" Defining the Logistic Regression Model"""

base_columns = [gender,native_country,education, occupation ,workclass, relationship, age_buckets]

crossed_columns = [education_x_occupation, age_x_education_x_occupation,
                   tf.feature_column.crossed_column(["native_country", "occupation"], hash_bucket_size=1000)]

model_dir = tmp.mkdtemp()

print("hello im in")

m =tf.estimator.LinearClassifier(model_dir=model_dir, feature_columns = base_columns + crossed_columns,
     optimizer=tf.train.FtrlOptimizer(
    learning_rate= 0.001,
    l1_regularization_strength= 1.0,
    l2_regularization_strength= 1.0))

print("finished with model")

""" Training the logistic classifier"""
m.train(
    input_fn=input_fn(train_file,num_epochs=None, shuffle=True), steps=100)

print("finished training")

results = m.evaluate(
    input_fn=input_fn(test_file, num_epochs=1, shuffle=False),steps=None)

print("model directory = %s" % model_dir)
for key in sorted(results):
  print("%s: %s" % (key, results[key]))
