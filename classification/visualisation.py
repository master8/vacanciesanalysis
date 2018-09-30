class Visualizer:

    @staticmethod
    def show_results(classifier_name, vec_method,
                     cross_val_accuracy, cross_val_f1,
                     train_accuracy, train_f1,
                     test_accuracy, test_f1):
        print()
        print(classifier_name + " " + vec_method)
        print()
        print("cross validation")
        print("accuracy:" + str(cross_val_accuracy))
        print("accuracy mean:" + str(cross_val_accuracy.mean()))
        print("f-score:" + str(cross_val_f1))
        print("f-score mean:" + str(cross_val_f1.mean()))
        print()
        print("Prediction train: ")
        print("accuracy: " + str(train_accuracy))
        print("f-score: " + str(train_f1))
        print()
        print("Prediction test: ")
        print("accuracy: " + str(test_accuracy))
        print("f-score: " + str(test_f1))
        print()
