#include <Std.h>
#include <Errors.h>
#include <CudaSupport.h>
#include <Definitions.h>

///////////////////////////////////////////////////////////////////////////////

void SelectSuitableCudaDevice( ostream& out, size_t rank )
{
	int deviceCount = 0;
	cudaCheck( cudaGetDeviceCount( &deviceCount ) );
	out << "Number of cuda devices: " << deviceCount << endl;
	out << endl;
	vector<int> suitableDevices;
	for( int i = 0; i < deviceCount; i++ ) {
		cudaDeviceProp deviceProp;
		cudaCheck( cudaGetDeviceProperties( &deviceProp, i ) );
		out << "Device#" << left << setw( 13 ) << i
			<< ": " << deviceProp.name << endl
			<< "Compute capability  : " << deviceProp.major
			<< "." << deviceProp.minor << endl
			<< "Total global memory : " << deviceProp.totalGlobalMem << endl
			<< endl;

		if( deviceProp.major == 2 /* && deviceProp.minor == 0 */ ) {
			suitableDevices.push_back( i );
		}
	}

	if( suitableDevices.empty() ) {
		throw CException( "Suitable cuda device was not found!" );
	}

	const int suitableDevice = suitableDevices[rank % suitableDevices.size()];
	cudaCheck( cudaSetDevice( suitableDevice ) );
	out << "Device#" << suitableDevice << " has been selected." << endl << endl;
}

///////////////////////////////////////////////////////////////////////////////
