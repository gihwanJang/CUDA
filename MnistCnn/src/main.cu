#include "cnn/cudaCnn.h"
//#include "cnn/cnn.h"

int main(int argc, char const *argv[])
{
	//timer
	DS_timer timer(4);

	timer.setTimerName(0, (char *)"total Learning");
    timer.setTimerName(1, (char *)"kernel Learning");
	timer.setTimerName(2, (char *)"total classify");
    timer.setTimerName(3, (char *)"kernel classify");

	cudaCNN(timer);
	//CNN(timer);

	timer.printTimer();
	return 0;
}

