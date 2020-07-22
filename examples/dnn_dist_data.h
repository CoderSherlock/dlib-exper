// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
   This is an example illustrating the use of the deep learning tools from the
   dlib C++ Library.  In it, we will train the venerable LeNet convolutional
   neural network to recognize hand written digits.  The network will take as
   input a small image and classify it as one of the 10 numeric digits between
   0 and 9.

   The specific network we will run is from the paper
   LeCun, Yann, et al. "Gradient-based learning applied to document recognition."
   Proceedings of the IEEE 86.11 (1998): 2278-2324.
   except that we replace the sigmoid non-linearities with rectified linear units. 

   These tools will use CUDA and cuDNN to drastically accelerate network
   training and testing.  CMake should automatically find them if they are
   installed and configure things appropriately.  If not, the program will
   still run but will be much slower to execute.
   */


#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include <chrono>
#include <csignal>

using namespace std;
using namespace dlib;
using std::chrono::system_clock;

template <
		typename data_type,
		typename label_type
>
class dataset {

private:
	std::vector<data_type> 			datas;
	std::vector<label_type> 		labels;

public:
	bool verbose;

	dataset () = default;
	dataset (std::vector<data_type> d, std::vector<label_type> l) {
		this->datas = d;
		this->labels = l;
	}
	dataset (int (*function)(char*, std::vector<data_type>&, std::vector<label_type>&), char *filename){
		function(filename, this->datas, this->labels);
	}

	std::vector<data_type> getData() {
		return this->datas;
	}

	std::vector<label_type> getLabel() {
		return this->labels;
	}

	dataset split(size_t start, size_t end) {

		std::vector<data_type> child_datas = this->datas;
		std::vector<label_type> child_labels = this->labels;

		child_datas.erase(child_datas.begin(), child_datas.begin() + start);
		child_datas.erase(child_datas.begin() + end - start, child_datas.end());
		child_labels.erase(child_labels.begin(), child_labels.begin() + start);
		child_labels.erase(child_labels.begin() + end - start, child_labels.end());

		if (verbose)
			std::cout << "Dataset size is" << child_datas.size() << std::endl;

		return dataset<data_type, label_type>(child_datas, child_labels);
	}

	dataset split_by_group(size_t groups, size_t index) {
		size_t group_size = this->datas.size() / groups;

		std::vector<data_type> child_datas = this->datas;
		std::vector<label_type> child_labels = this->labels;

		child_datas.erase(child_datas.begin(), child_datas.begin() + index * group_size);
		child_datas.erase(child_datas.begin() + group_size, child_datas.end());
		child_labels.erase(child_labels.begin(), child_labels.begin() + index * group_size);
		child_labels.erase(child_labels.begin() + group_size, child_labels.end());

		return dataset<data_type, label_type>(child_datas, child_labels);
	}

	template <
			typename net_type
	>
	double accuracy(net_type net){
		auto epoch_time = system_clock::now();  // HPZ: Counting
		std::vector<label_type> predicted_labels = net(this->datas);
		int num_right = 0;
		int num_wrong = 0;
		// And then let's see if it classified them correctly.
		for (size_t i = 0; i < this->datas.size(); ++i)
		{
			if (predicted_labels[i] == this->labels[i])
				++num_right;
			else {
				++num_wrong;
				// if (predicted_labels[i] == 72 && this->labels[i] != 72) {
				// 	std::cout << "!!!!!!!!! " << i << " !!!!!!!!!!!" << std::endl;
				// }
			}
		}
		cout << "testing num_right: " << num_right << endl;
		cout << "testing num_wrong: " << num_wrong << endl;
		cout << "testing accuracy:  " << num_right/(double)(num_right+num_wrong) << endl;
		std::cout << "Time for Validation is "
				  << std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - epoch_time).count() << std::endl;   // HPZ: Counting
		return num_right/(double)(num_right+num_wrong);
	}

};

int load_mnist_training_data(char* dataset, std::vector<matrix<unsigned char>> &data, std::vector<unsigned long> &label) {
	// try{
		std::vector<matrix < unsigned char>> training_images;
		std::vector<unsigned long> training_labels;
		std::vector<matrix < unsigned char>> testing_images;
		std::vector<unsigned long> testing_labels;
		load_mnist_dataset(dataset, training_images, training_labels, testing_images, testing_labels);

		data = training_images;
		label = training_labels;
	// } catch(...) {
	// 	std::cerr << "Something wrong with loading training data and labels"	<< std::endl;
	// 	return 0;
	// }
	return 1;

};

int load_mnist_testing_data(char* dataset, std::vector<matrix<unsigned char>> &data, std::vector<unsigned long> &label) {
    try{
        std::vector<matrix < unsigned char>> training_images;
        std::vector<unsigned long> training_labels;
        std::vector<matrix < unsigned char>> testing_images;
        std::vector<unsigned long> testing_labels;
        load_mnist_dataset(dataset, training_images, training_labels, testing_images, testing_labels);

        data = testing_images;
        label = testing_labels;
    } catch(...) {
        std::cerr << "Something wrong with loading testing data and labels"	<< std::endl;
        return 0;
    }
    return 1;
};

