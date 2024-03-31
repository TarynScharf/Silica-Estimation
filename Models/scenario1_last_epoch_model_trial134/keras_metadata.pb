
�Proot"_tf_keras_sequential*�P{"name": "sequential_134", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": false, "class_name": "Sequential", "config": {"name": "sequential_134", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float64", "sparse": false, "ragged": false, "name": "dense_631_input"}}, {"class_name": "Dense", "config": {"name": "dense_631", "trainable": true, "dtype": "float32", "units": 120, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_497", "trainable": true, "dtype": "float32", "alpha": 0.009999999776482582}}, {"class_name": "Dropout", "config": {"name": "dropout_497", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_632", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_498", "trainable": true, "dtype": "float32", "alpha": 0.009999999776482582}}, {"class_name": "Dropout", "config": {"name": "dropout_498", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_633", "trainable": true, "dtype": "float32", "units": 110, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_499", "trainable": true, "dtype": "float32", "alpha": 0.009999999776482582}}, {"class_name": "Dropout", "config": {"name": "dropout_499", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_634", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_500", "trainable": true, "dtype": "float32", "alpha": 0.009999999776482582}}, {"class_name": "Dropout", "config": {"name": "dropout_500", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_635", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 29, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}, "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [32, 6]}, "float64", "dense_631_input"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [32, 6]}, "float64", "dense_631_input"]}, "keras_version": "2.10.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_134", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float64", "sparse": false, "ragged": false, "name": "dense_631_input"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_631", "trainable": true, "dtype": "float32", "units": 120, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_497", "trainable": true, "dtype": "float32", "alpha": 0.009999999776482582}, "shared_object_id": 5}, {"class_name": "Dropout", "config": {"name": "dropout_497", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}, "shared_object_id": 6}, {"class_name": "Dense", "config": {"name": "dense_632", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 9}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_498", "trainable": true, "dtype": "float32", "alpha": 0.009999999776482582}, "shared_object_id": 11}, {"class_name": "Dropout", "config": {"name": "dropout_498", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}, "shared_object_id": 12}, {"class_name": "Dense", "config": {"name": "dense_633", "trainable": true, "dtype": "float32", "units": 110, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 15}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_499", "trainable": true, "dtype": "float32", "alpha": 0.009999999776482582}, "shared_object_id": 17}, {"class_name": "Dropout", "config": {"name": "dropout_499", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}, "shared_object_id": 18}, {"class_name": "Dense", "config": {"name": "dense_634", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 21}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 22}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_500", "trainable": true, "dtype": "float32", "alpha": 0.009999999776482582}, "shared_object_id": 23}, {"class_name": "Dropout", "config": {"name": "dropout_500", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}, "shared_object_id": 24}, {"class_name": "Dense", "config": {"name": "dense_635", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 25}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 27}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 28}]}}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}, "shared_object_id": 30}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mape_metric", "dtype": "float32", "fn": "mape_metric"}, "shared_object_id": 31}, {"class_name": "MeanMetricWrapper", "config": {"name": "smape_metric", "dtype": "float32", "fn": "smape_metric"}, "shared_object_id": 32}, {"class_name": "MeanMetricWrapper", "config": {"name": "concordance_cc", "dtype": "float32", "fn": "concordance_cc"}, "shared_object_id": 33}, {"class_name": "MeanMetricWrapper", "config": {"name": "pearson_cc", "dtype": "float32", "fn": "pearson_cc"}, "shared_object_id": 34}, {"class_name": "Addons>RSquare", "config": {"name": "r_square", "dtype": "float32", "multioutput": "uniform_average"}, "shared_object_id": 35}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Nadam", "config": {"name": "Nadam", "learning_rate": 0.0010000000474974513, "decay": 0.004, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07}}}}2
�root.layer_with_weights-0"_tf_keras_layer*�{"name": "dense_631", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_631", "trainable": true, "dtype": "float32", "units": 120, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}, "shared_object_id": 36}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 6]}}2
�root.layer-1"_tf_keras_layer*�{"name": "leaky_re_lu_497", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_497", "trainable": true, "dtype": "float32", "alpha": 0.009999999776482582}, "shared_object_id": 5, "build_input_shape": {"class_name": "TensorShape", "items": [32, 120]}}2
�root.layer-2"_tf_keras_layer*�{"name": "dropout_497", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dropout", "config": {"name": "dropout_497", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}, "shared_object_id": 6, "build_input_shape": {"class_name": "TensorShape", "items": [32, 120]}}2
�root.layer_with_weights-1"_tf_keras_layer*�{"name": "dense_632", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_632", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 9}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 120}}, "shared_object_id": 37}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 120]}}2
�root.layer-4"_tf_keras_layer*�{"name": "leaky_re_lu_498", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_498", "trainable": true, "dtype": "float32", "alpha": 0.009999999776482582}, "shared_object_id": 11, "build_input_shape": {"class_name": "TensorShape", "items": [32, 80]}}2
�root.layer-5"_tf_keras_layer*�{"name": "dropout_498", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dropout", "config": {"name": "dropout_498", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}, "shared_object_id": 12, "build_input_shape": {"class_name": "TensorShape", "items": [32, 80]}}2
�root.layer_with_weights-2"_tf_keras_layer*�{"name": "dense_633", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_633", "trainable": true, "dtype": "float32", "units": 110, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 15}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 80}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 80]}}2
�root.layer-7"_tf_keras_layer*�{"name": "leaky_re_lu_499", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_499", "trainable": true, "dtype": "float32", "alpha": 0.009999999776482582}, "shared_object_id": 17, "build_input_shape": {"class_name": "TensorShape", "items": [32, 110]}}2
�	root.layer-8"_tf_keras_layer*�{"name": "dropout_499", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dropout", "config": {"name": "dropout_499", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}, "shared_object_id": 18, "build_input_shape": {"class_name": "TensorShape", "items": [32, 110]}}2
�
root.layer_with_weights-3"_tf_keras_layer*�{"name": "dense_634", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_634", "trainable": true, "dtype": "float32", "units": 80, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 21}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 110}}, "shared_object_id": 39}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 110]}}2
�
�
�
��root.keras_api.metrics.0"_tf_keras_metric*�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 41}2
��root.keras_api.metrics.1"_tf_keras_metric*�{"class_name": "MeanMetricWrapper", "name": "mape_metric", "dtype": "float32", "config": {"name": "mape_metric", "dtype": "float32", "fn": "mape_metric"}, "shared_object_id": 31}2
��root.keras_api.metrics.2"_tf_keras_metric*�{"class_name": "MeanMetricWrapper", "name": "smape_metric", "dtype": "float32", "config": {"name": "smape_metric", "dtype": "float32", "fn": "smape_metric"}, "shared_object_id": 32}2
��root.keras_api.metrics.3"_tf_keras_metric*�{"class_name": "MeanMetricWrapper", "name": "concordance_cc", "dtype": "float32", "config": {"name": "concordance_cc", "dtype": "float32", "fn": "concordance_cc"}, "shared_object_id": 33}2
��root.keras_api.metrics.4"_tf_keras_metric*�{"class_name": "MeanMetricWrapper", "name": "pearson_cc", "dtype": "float32", "config": {"name": "pearson_cc", "dtype": "float32", "fn": "pearson_cc"}, "shared_object_id": 34}2
��root.keras_api.metrics.5"_tf_keras_metric*�{"class_name": "Addons>RSquare", "name": "r_square", "dtype": "float32", "config": {"name": "r_square", "dtype": "float32", "multioutput": "uniform_average"}, "shared_object_id": 35}2