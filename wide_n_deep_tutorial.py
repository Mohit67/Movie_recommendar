# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import urllib

import pandas as pd
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "", "Base directory for output models.")
flags.DEFINE_string("model_type", "wide_n_deep",
                    "Valid model types: {'wide', 'deep', 'wide_n_deep'}.")
flags.DEFINE_integer("train_steps", 700, "Number of training steps.")
flags.DEFINE_string(
    "train_data",
    "",
    "Path to the training data.")
flags.DEFINE_string(
    "test_data",
    "",
    "Path to the test data.")

#COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
#           "marital_status", "occupation", "relationship", "race", "gender",
#           "capital_gain", "capital_loss", "hours_per_week", "native_country",
#           "income_bracket"]

#LABEL_COLUMN = "label"
#CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
#                       "relationship", "race", "gender", "native_country"]
#CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
#                      "hours_per_week"]

LABEL_COLUMN = "label"

COLUMNS = ["User_id", "Movie_id", "Rating", "Timestamp", "Action", "Adventure",
			"Animation", "Children", "Comedy", "Crime", "Documentary", "Drama",
			"Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
			"Sci-fi", "Thriller", "War", "Western", "Zip-code"]

#check how putting "Rating" in categorical changes the accuracy
CATEGORICAL_COLUMNS = ["User_id", "Action", "Adventure", "Animation", "Children",
			"Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
			"Musical", "Mystery", "Romance", "Sci-fi", "Thriller", "War", "Western", "Zip-code"]

#CONTINUOUS_COLUMNS = ["Timestamp"]


#def maybe_download():
#  """May be downloads training data and returns train and test file names."""
#  if FLAGS.train_data:
#    train_file_name = FLAGS.train_data
#  else:
#    train_file = tempfile.NamedTemporaryFile(delete=False)
#    urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)  # pylint: disable=line-too-long
#    train_file_name = train_file.name
#    train_file.close()
#    print("Training data is downloaded to %s" % train_file_name)
#
#  if FLAGS.test_data:
#    test_file_name = FLAGS.test_data
#  else:
#    test_file = tempfile.NamedTemporaryFile(delete=False)
#    urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)  # pylint: disable=line-too-long
#    test_file_name = test_file.name
#    test_file.close()
#    print("Test data is downloaded to %s" % test_file_name)
#
#  return train_file_name, test_file_name


def build_estimator(model_dir):
  """Build an estimator."""

  action = tf.contrib.layers.sparse_column_with_keys(column_name="Action",
                                                     keys=["0", "1"])
  adventure = tf.contrib.layers.sparse_column_with_keys(column_name="Adventure",
                                                     keys=["0", "1"]) 
  animation = tf.contrib.layers.sparse_column_with_keys(column_name="Animation",
                                                     keys=["0", "1"])
  children = tf.contrib.layers.sparse_column_with_keys(column_name="Children",
                                                     keys=["0", "1"])
  comedy = tf.contrib.layers.sparse_column_with_keys(column_name="Comedy",
                                                     keys=["0", "1"])
  crime = tf.contrib.layers.sparse_column_with_keys(column_name="Crime",
                                                     keys=["0", "1"])
  documentary = tf.contrib.layers.sparse_column_with_keys(column_name="Documentary",
                                                     keys=["0", "1"])
  drama = tf.contrib.layers.sparse_column_with_keys(column_name="Drama",
                                                     keys=["0", "1"])
  fantasy = tf.contrib.layers.sparse_column_with_keys(column_name="Fantasy",
                                                     keys=["0", "1"])
  film_noir = tf.contrib.layers.sparse_column_with_keys(column_name="Film-Noir",
                                                     keys=["0", "1"])
  horror = tf.contrib.layers.sparse_column_with_keys(column_name="Horror",
                                                     keys=["0", "1"]) 
  musical = tf.contrib.layers.sparse_column_with_keys(column_name="Musical",
                                                     keys=["0", "1"])
  mystery = tf.contrib.layers.sparse_column_with_keys(column_name="Mystery",
                                                     keys=["0", "1"])
  romance = tf.contrib.layers.sparse_column_with_keys(column_name="Romance",
                                                     keys=["0", "1"])
  sci_fi = tf.contrib.layers.sparse_column_with_keys(column_name="Sci-fi",
                                                     keys=["0", "1"])
  thriller = tf.contrib.layers.sparse_column_with_keys(column_name="Thriller",
                                                     keys=["0", "1"])
  war = tf.contrib.layers.sparse_column_with_keys(column_name="War",
                                                     keys=["0", "1"])
  western = tf.contrib.layers.sparse_column_with_keys(column_name="Western",
                                                     keys=["0", "1"])

#  rating = tf.contrib.layers.sparse_column_with_keys(column_name="Rating",
#                                                     keys=["1", "2", "3", "4", "5"])

  user_id = tf.contrib.layers.sparse_column_with_hash_bucket(
      "User_id", hash_bucket_size=1000)	
  zip_code = tf.contrib.layers.sparse_column_with_hash_bucket(
      "Zip-code", hash_bucket_size=1000)
#  action = tf.contrib.layers.sparse_column_with_hash_bucket(
#      "Action", hash_bucket_size=2)
#  adventure = tf.contrib.layers.sparse_column_with_hash_bucket(
#      "Adventure", hash_bucket_size=2)
#  animation = tf.contrib.layers.sparse_column_with_hash_bucket(
#      "Animation", hash_bucket_size=2)
#  children = tf.contrib.layers.sparse_column_with_hash_bucket(
#      "Children", hash_bucket_size=2)
#  comedy = tf.contrib.layers.sparse_column_with_hash_bucket(
#      "Comedy", hash_bucket_size=2)
#  crime = tf.contrib.layers.sparse_column_with_hash_bucket(
#      "Crime", hash_bucket_size=2)
#  documentary = tf.contrib.layers.sparse_column_with_hash_bucket(
#      "Documentary", hash_bucket_size=2)
#  drama = tf.contrib.layers.sparse_column_with_hash_bucket(
#      "Drama", hash_bucket_size=2)
#  fantasy = tf.contrib.layers.sparse_column_with_hash_bucket(
#      "Fantasy", hash_bucket_size=2)
#  film_noir = tf.contrib.layers.sparse_column_with_hash_bucket(
#      "Film-Noir", hash_bucket_size=2)
#  horror = tf.contrib.layers.sparse_column_with_hash_bucket(
#      "Horror", hash_bucket_size=2)
#  musical = tf.contrib.layers.sparse_column_with_hash_bucket(
#      "Musical", hash_bucket_size=2)
#  mystery = tf.contrib.layers.sparse_column_with_hash_bucket(
#      "Mystery", hash_bucket_size=2)
#  romance = tf.contrib.layers.sparse_column_with_hash_bucket(
#      "Romance", hash_bucket_size=2)
#  sci_fi = tf.contrib.layers.sparse_column_with_hash_bucket(
#      "Sci-fi", hash_bucket_size=2)
#  thriller = tf.contrib.layers.sparse_column_with_hash_bucket(
#      "Thriller", hash_bucket_size=2)
#  war = tf.contrib.layers.sparse_column_with_hash_bucket(
#      "War", hash_bucket_size=2)
#  western = tf.contrib.layers.sparse_column_with_hash_bucket(
#      "Western", hash_bucket_size=2)

                                                                                                                                                                                             
  # Sparse base columns.
#  gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender",
#                                                     keys=["female", "male"])
#  race = tf.contrib.layers.sparse_column_with_keys(column_name="race",
#                                                   keys=["Amer-Indian-Eskimo",
#                                                         "Asian-Pac-Islander",
#                                                         "Black", "Other",
#                                                         "White"])
#
#  education = tf.contrib.layers.sparse_column_with_hash_bucket(
#      "education", hash_bucket_size=1000)
#  marital_status = tf.contrib.layers.sparse_column_with_hash_bucket(
#      "marital_status", hash_bucket_size=100)
#  relationship = tf.contrib.layers.sparse_column_with_hash_bucket(
#      "relationship", hash_bucket_size=100)
#  workclass = tf.contrib.layers.sparse_column_with_hash_bucket(
#      "workclass", hash_bucket_size=100)
#  occupation = tf.contrib.layers.sparse_column_with_hash_bucket(
#      "occupation", hash_bucket_size=1000)
#  native_country = tf.contrib.layers.sparse_column_with_hash_bucket(
#      "native_country", hash_bucket_size=1000)



#  timestamp = tf.contrib.layers.real_valued_column("Timestamp")



  # Continuous base columns.
#  age = tf.contrib.layers.real_valued_column("age")
#  education_num = tf.contrib.layers.real_valued_column("education_num")
#  capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
#  capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
#  hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

  # Transformations.
#  age_buckets = tf.contrib.layers.bucketized_column(age,
#                                                    boundaries=[
#                                                        18, 25, 30, 35, 40, 45,
#                                                        50, 55, 60, 65
#                                                    ])

  crossed_correlated_genres = tf.contrib.layers.crossed_column(
  								[action, adventure, animation, children, comedy,
  								 crime, documentary, drama, fantasy, film_noir,
  								 horror, musical, mystery, romance, sci_fi, thriller,
  								 war, western], hash_bucket_size=int(1e4))
  # zip_code_x_genres = tf.contrib.layers.crossed_column(
  # 								[zip_code, crossed_correlated_genres], hash_bucket_size=int(1e4))
  zip_code_x_genres = tf.contrib.layers.crossed_column(
  								[zip_code, crossed_correlated_genres], hash_bucket_size=int(1e4)) 

  user_x_zip = tf.contrib.layers.crossed_column([user_id, zip_code],
                                                   hash_bucket_size=int(1e4))

  wide_columns = [user_id, zip_code, user_x_zip, zip_code_x_genres,
  					tf.contrib.layers.crossed_column([user_id, crossed_correlated_genres], 
  																								 hash_bucket_size=int(1e4)),
  				  tf.contrib.layers.crossed_column([user_id, zip_code_x_genres],
                                                   hash_bucket_size=int(1e4)),
  				  tf.contrib.layers.crossed_column([user_x_zip, zip_code_x_genres],
                                                   hash_bucket_size=int(1e4))]  				  
# try for user-zip and zip-genre

  deep_columns = [
      tf.contrib.layers.embedding_column(user_id, dimension=8),  	  
      tf.contrib.layers.embedding_column(zip_code, dimension=8),  	  
      # tf.contrib.layers.embedding_column(action, dimension=8),  	  
      # tf.contrib.layers.embedding_column(adventure, dimension=8),  	  
      # tf.contrib.layers.embedding_column(animation, dimension=8),  	  
      # tf.contrib.layers.embedding_column(children, dimension=8),  	  
      # tf.contrib.layers.embedding_column(comedy, dimension=8),  	  
      # tf.contrib.layers.embedding_column(crime, dimension=8),  	  
      # tf.contrib.layers.embedding_column(documentary, dimension=8),  	  
      # tf.contrib.layers.embedding_column(drama, dimension=8),  	  
      # tf.contrib.layers.embedding_column(fantasy, dimension=8),  	  
      # tf.contrib.layers.embedding_column(film_noir, dimension=8),  	  
      # tf.contrib.layers.embedding_column(horror, dimension=8),  	  
      # tf.contrib.layers.embedding_column(musical, dimension=8),  	  
      # tf.contrib.layers.embedding_column(mystery, dimension=8),  	  
      # tf.contrib.layers.embedding_column(romance, dimension=8),  	  
      # tf.contrib.layers.embedding_column(sci_fi, dimension=8),  	  
      # tf.contrib.layers.embedding_column(thriller, dimension=8),  	  
      # tf.contrib.layers.embedding_column(war, dimension=8),  	  
      # tf.contrib.layers.embedding_column(western, dimension=8),  	  
      # timestamp,
  ]

  # Wide columns and deep columns.
#  wide_columns = [gender, native_country, education, occupation, workclass,
#                  marital_status, relationship, age_buckets,
#                  tf.contrib.layers.crossed_column([education, occupation],
#                                                   hash_bucket_size=int(1e4)),
#                  tf.contrib.layers.crossed_column(
#                      [age_buckets, race, occupation],
#                      hash_bucket_size=int(1e6)),
#                  tf.contrib.layers.crossed_column([native_country, occupation],
#                                                   hash_bucket_size=int(1e4))]
#  deep_columns = [
#      tf.contrib.layers.embedding_column(workclass, dimension=8),
#      tf.contrib.layers.embedding_column(education, dimension=8),
#      tf.contrib.layers.embedding_column(marital_status,
#                                         dimension=8),
#      tf.contrib.layers.embedding_column(gender, dimension=8),
#      tf.contrib.layers.embedding_column(relationship, dimension=8),
#      tf.contrib.layers.embedding_column(race, dimension=8),
#      tf.contrib.layers.embedding_column(native_country,
#                                         dimension=8),
#      tf.contrib.layers.embedding_column(occupation, dimension=8),
#      age,
#      education_num,
#      capital_gain,
#      capital_loss,
#      hours_per_week,
#  ]

  if FLAGS.model_type == "wide":
    m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                          feature_columns=wide_columns)
  elif FLAGS.model_type == "deep":
    m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                       feature_columns=deep_columns,
                                       hidden_units=[100, 50])
  else:
    m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50],
        n_classes=5)
  return m


def input_fn(df):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.


#  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}


  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}

  # Merges the two dictionaries into one.
  # feature_cols = dict(continuous_cols)
  # feature_cols.update(categorical_cols)
  feature_cols = dict(categorical_cols)


  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label

#def myfunc(x):
#	mytensor = []
#	mytensor.append(x["Rating"]*x["Action"])
#	mytensor.append(x["Rating"]*x["Adventure"])
#	mytensor.append(x["Rating"]*x["Animation"])
#	mytensor.append(x["Rating"]*x["Children"])
#	mytensor.append(x["Rating"]*x["Comedy"])
#	mytensor.append(x["Rating"]*x["Crime"])
#	mytensor.append(x["Rating"]*x["Documentary"])
#	mytensor.append(x["Rating"]*x["Drama"])
#	mytensor.append(x["Rating"]*x["Fantasy"])
#	mytensor.append(x["Rating"]*x["Film-Noir"])
#	mytensor.append(x["Rating"]*x["Horror"])
#	mytensor.append(x["Rating"]*x["Musical"])
#	mytensor.append(x["Rating"]*x["Mystery"])
#	mytensor.append(x["Rating"]*x["Romance"])
#	mytensor.append(x["Rating"]*x["Sci-fi"])
#	mytensor.append(x["Rating"]*x["Thriller"])
#	mytensor.append(x["Rating"]*x["War"])
#	mytensor.append(x["Rating"]*x["Western"])
#	return mytensor

def train_and_eval():
  """Train and evaluate the model."""
#  train_file_name = "datafile.csv"
#  test_file_name = "datafile.txt"

  df_train = pd.read_csv("trainfile.csv", dtype={
  	'User_id': 	 str,
  	"Action": 	 str,
  	"Adventure": str, 
  	"Animation": str, 
  	"Children":  str,
	"Comedy":    str, 
	"Crime": 	 str, 
	"Drama":     str, 
	"Fantasy":   str, 
	"Film-Noir": str, 
	"Horror":    str,
	"Musical":   str, 
	"Mystery":   str, 
	"Romance":   str, 
	"Sci-fi":    str, 
	"Thriller":  str, 
	"War":       str, 
	"Western":   str, 
	"Zip-code":  str,
	"Documentary": str, 
  	}, names=COLUMNS, skipinitialspace=True)

  df_test = pd.read_csv("testfile.csv", dtype={
  	'User_id': 	 str,
  	"Action": 	 str,
  	"Adventure": str, 
  	"Animation": str, 
  	"Children":  str,
	"Comedy":    str, 
	"Crime": 	 str, 
	"Drama":     str, 
	"Fantasy":   str, 
	"Film-Noir": str, 
	"Horror":    str,
	"Musical":   str, 
	"Mystery":   str, 
	"Romance":   str, 
	"Sci-fi":    str, 
	"Thriller":  str, 
	"War":       str, 
	"Western":   str, 
	"Zip-code":  str,
	"Documentary": str, 
  	}, names=COLUMNS, skipinitialspace=True)

#  df_test = pd.read_csv(
#      tf.gfile.Open(test_file_name),
#      names=COLUMNS,
#      skipinitialspace=True)
#      skiprows=1)

#the apply method applies the function(myfunc) to each row(axis=1) or column(axis=0)
#apply sends each series(here the row series as parameter)
#  df_train[LABEL_COLUMN] = df_train.apply(myfunc, axis=1)	

  df_train[LABEL_COLUMN] = df_train["Rating"].apply(lambda x: 1 if(x == 1) else 2 if(x == 2) else 3 if(x == 3) else 4 if(x == 4) else 5)
  # df_train[LABEL_COLUMN] = df_train["Rating"]
  df_test[LABEL_COLUMN] = df_test["Rating"]

#  for col in CATEGORICAL_COLUMNS:
#  	print(df_train[col])

#  df_train[LABEL_COLUMN] = (
#      df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
#  df_test[LABEL_COLUMN] = (
#      df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

#  model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
#  print("model directory = %s" % model_dir)


  model_dir = "model_directory"

  m = build_estimator(model_dir)
  m.fit(input_fn=lambda: input_fn(df_train), steps=FLAGS.train_steps)

  # results = m.evaluate(input_fn=lambda: input_fn(df_test))
  # print("results")
  # print(len(results))
  # print(results)


#  ans = m.predict(df_test)
#  print('Predictions: {}'.format(str(ans)))

  # result1 = m.predict_proba(input_fn=lambda: input_fn(df_test))
  # print(result1)


  results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
  for key in sorted(results):
	  print("%s: %s" % (key, results[key]))

def main(_):
  train_and_eval()


if __name__ == "__main__":
  tf.app.run()

