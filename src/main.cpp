#include <Std.h>

#include <MpiSupport.h>
#include <CudaSupport.h>
#include <Definitions.h>
#include <MathObjects.h>
#include <CudaObjects.h>
#include <IterationCallback.h>
#include <kernel.h>

///////////////////////////////////////////////////////////////////////////////

class CExchangeDefinition {
public:
	CExchangeDefinition( size_t rank,
			const CMatrixPart& sendPart,
			const CMatrixPart& recvPart ) :
		rank( rank ),
		sendPart( sendPart ),
		recvPart( recvPart )
	{
		sendBuffer.resize( sendPart.Size() );
		recvBuffer.resize( recvPart.Size() );
	}

	const CMatrixPart& SendPart() const { return sendPart; }
	const CMatrixPart& RecvPart() const { return recvPart; }

	void DoExchange( cudaMatrix& matrix );
	void Wait( cudaMatrix& matrix );

private:
	size_t rank;

	CMatrixPart sendPart;
	MPI_Request sendRequest;
	vector<NumericType> sendBuffer;

	CMatrixPart recvPart;
	MPI_Request recvRequest;
	vector<NumericType> recvBuffer;
};

///////////////////////////////////////////////////////////////////////////////

void CExchangeDefinition::DoExchange( cudaMatrix& matrix )
{
	matrix.GetPart( sendPart, sendBuffer );
	MpiCheck( MPI_Isend( sendBuffer.data(), sendBuffer.size(),
		MpiNumericType, rank, 0, MPI_COMM_WORLD, &sendRequest ), "MPI_Isend" );

	MpiCheck( MPI_Irecv( recvBuffer.data(), recvBuffer.size(),
		MpiNumericType, rank, 0, MPI_COMM_WORLD, &recvRequest ), "MPI_Irecv" );
}

void CExchangeDefinition::Wait( cudaMatrix& matrix )
{
	MpiCheck( MPI_Wait( &recvRequest, MPI_STATUS_IGNORE ), "MPI_Wait" );
	matrix.SetPart( recvPart, recvBuffer );

	MpiCheck( MPI_Wait( &sendRequest, MPI_STATUS_IGNORE ), "MPI_Wait" );
}

///////////////////////////////////////////////////////////////////////////////

class CExchangeDefinitions : public vector<CExchangeDefinition> {
public:
	CExchangeDefinitions() {}

	void Exchange( cudaMatrix& matrix )
	{
		for( vector<CExchangeDefinition>::iterator i = begin(); i != end(); ++i ) {
			i->DoExchange( matrix );
		}
		// Сначала начинаем все асинхронные операции, затем ждём.
		for( vector<CExchangeDefinition>::iterator i = begin(); i != end(); ++i ) {
			i->Wait( matrix );
		}
	}
};

///////////////////////////////////////////////////////////////////////////////

void GetBeginEndPoints( const size_t numberOfPoints, const size_t numberOfBlocks,
	const size_t blockIndex, size_t& beginPoint, size_t& endPoint )
{
	const size_t objectsPerProcess = numberOfPoints / numberOfBlocks;
	const size_t additionalPoints = numberOfPoints % numberOfBlocks;
	beginPoint = objectsPerProcess * blockIndex + min( blockIndex, additionalPoints );
	endPoint = beginPoint + objectsPerProcess;
	if( blockIndex < additionalPoints ) {
		endPoint++;
	}
}

///////////////////////////////////////////////////////////////////////////////

NumericType TotalError( const CMatrix& p, const CUniformGrid& grid )
{
	NumericType error = 0;
	for( size_t x = 0; x < p.SizeX(); x++ ) {
		for( size_t y = 0; y < p.SizeY(); y++ ) {
			error = max( error, abs( Phi( grid.X[x], grid.Y[y] ) - p( x, y ) ) );
		}
	}
	return error;
}

///////////////////////////////////////////////////////////////////////////////

void DumpMatrix( const CMatrix& matrix, const CUniformGrid& grid, ostream& output )
{
	for( size_t x = 0; x < matrix.SizeX(); x++ ) {
		for( size_t y = 0; y < matrix.SizeY(); y++ ) {
			output << grid.X[x] << '\t' << grid.Y[y] << '\t' << matrix( x, y ) << endl;
		}
	}
}

///////////////////////////////////////////////////////////////////////////////

class CBaseProgram {
public:
	// Последовательная реализация.
	static void Run( const size_t pointsX, const size_t pointsY, const CArea& area,
		IIterationCallback& callback, const string& dumpFilename = "" );

protected:
	CUniformGrid grid;
	CMatrix p;
	NumericType difference;
	// cuda objects
	dim3 cudaGridDim;
	cudaMatrix cudaP;
	cudaMatrix cudaR;
	cudaMatrix cudaG;
	cudaMatrix cudaBuffer; // auxiliary buffer
	cudaUniformGrid cudaGrid;

	CBaseProgram() {}
	void Initialze();

private:
	void run( IIterationCallback& callback, const string& dumpFilename );
};

void CBaseProgram::Initialze()
{
	// cuda grid dim (вычитаем 2 т.к. значения на границах вычислять не нужно).
	cudaGridDim.x = DivUp( grid.X.Size() - 2, BlockDimX );
	cudaGridDim.y = DivUp( grid.Y.Size() - 2, BlockDimY );

	// cuda grid
	cudaGrid.X.Allocate( grid.X );
	cudaGrid.Y.Allocate( grid.Y );

	// cuda auxiliary buffer
	{
		CMatrix buffer( cudaGridDim.x * cudaGridDim.y, 2 );
		cudaBuffer.Allocate( buffer );
	}

	difference = numeric_limits<NumericType>::max();
}

void CBaseProgram::Run( const size_t pointsX, const size_t pointsY, const CArea& area,
	IIterationCallback& callback, const string& dumpFilename )
{
	// Инициализируем grid.
	CBaseProgram program;
	program.grid.X.Init( area.X0, area.Xn, pointsX );
	program.grid.Y.Init( area.Y0, area.Yn, pointsY );

	// Выполняем инициализацию.
	program.Initialze();

	// Запускаем последовательную реализацию.
	program.run( callback, dumpFilename );
}

void CBaseProgram::run( IIterationCallback& callback, const string& dumpFilename )
{
	// Выполняем нулевую итерацию (инициализацию).
	if( !callback.BeginIteration() ) {
		return;
	}

	CMatrix p( grid.X.Size(), grid.Y.Size() );
	for( size_t x = 0; x < p.SizeX(); x++ ) {
		p( x, 0 ) = Phi( grid.X[x], grid.Y[0] );
		p( x, p.SizeY() - 1 ) = Phi( grid.X[x], grid.Y[p.SizeY() - 1] );
	}
	for( size_t y = 1; y < p.SizeY() - 1; y++ ) {
		p( 0, y ) = Phi( grid.X[0], grid.Y[y] );
		p( p.SizeX() - 1, y ) = Phi( grid.X[p.SizeX() - 1], grid.Y[y] );
	}
	cudaP.Allocate( p );
	callback.EndIteration( difference );

	// Выполняем первую итерацию.
	if( !callback.BeginIteration() ) {
		return;
	}
	{
		//__debugbreak();
		CMatrix r( grid.X.Size(), grid.Y.Size() );
		cudaR.Allocate( r );

		CalcR( cudaGridDim, cudaP, cudaGrid, cudaR );
		const CFraction tau = CalcTau( cudaGridDim, cudaR, cudaR, cudaGrid, cudaBuffer );
		difference = CalcP( cudaGridDim, cudaR, tau.Value(), cudaP, cudaBuffer );

		cudaR.Dump( r );
		cudaG.Allocate( r );
	}
	callback.EndIteration( difference );

	// Выполняем остальные итерации.
	while( callback.BeginIteration() ) {
		CalcR( cudaGridDim, cudaP, cudaGrid, cudaR );
		const CFraction alpha = CalcAlpha( cudaGridDim, cudaR, cudaG, cudaGrid, cudaBuffer );
		CalcG( cudaGridDim, cudaR, alpha.Value(), cudaG );
		const CFraction tau = CalcTau( cudaGridDim, cudaR, cudaG, cudaGrid, cudaBuffer );
		difference = CalcP( cudaGridDim, cudaG, tau.Value(), cudaP, cudaBuffer );

		callback.EndIteration( difference );
	}

	cudaP.Dump( p );
	cout << "Total error: " << TotalError( p, grid ) << endl;

	if( !dumpFilename.empty() ) {
		ofstream outputFile( dumpFilename.c_str() );
		DumpMatrix( p, grid, outputFile );
	}
}

///////////////////////////////////////////////////////////////////////////////

class CProgram : private CBaseProgram {
public:
	static void Run( size_t pointsX, size_t pointsY, const CArea& area,
		IIterationCallback& callback );

private:
	const size_t numberOfProcesses;
	const size_t rank;
	const size_t pointsX;
	const size_t pointsY;
	size_t processesX;
	size_t processesY;
	size_t rankX;
	size_t rankY;
	size_t beginX;
	size_t endX;
	size_t beginY;
	size_t endY;
	CExchangeDefinitions exchangeDefinitions;

	CProgram( size_t pointsX, size_t pointsY, const CArea& area );

	bool hasLeftNeighbor() const { return ( rankX > 0 ); }
	bool hasRightNeighbor() const { return ( rankX < ( processesX - 1 ) ); }
	bool hasTopNeighbor() const { return ( rankY > 0 ); }
	bool hasBottomNeighbor() const { return ( rankY < ( processesY - 1 ) ); }
	size_t rankByXY( size_t x, size_t y ) const { return ( y * processesX + x ); }

	void setProcessXY();
	void setExchangeDefinitions();
	void allReduceFraction( CFraction& fraction );
	void allReduceDifference();
	void iteration0();
	void iteration1();
	void iteration2();
};

///////////////////////////////////////////////////////////////////////////////

void CProgram::Run( size_t pointsX, size_t pointsY, const CArea& area,
	IIterationCallback& callback )
{
	CProgram program( pointsX, pointsY, area );

	// Выполняем нулевую итерацию (инициализацию).
	if( !callback.BeginIteration() ) {
		return;
	}
	program.iteration0();
	callback.EndIteration( program.difference );

	// Выполняем первую итерацию.
	if( !callback.BeginIteration() ) {
		return;
	}
	program.iteration1();
	callback.EndIteration( program.difference );

	// Выполняем остальные итерации.
	while( callback.BeginIteration() ) {
		program.iteration2();
		callback.EndIteration( program.difference );
	}
}

CProgram::CProgram( size_t pointsX, size_t pointsY, const CArea& area ) :
	numberOfProcesses( CMpiSupport::NumberOfProccess() ),
	rank( CMpiSupport::Rank() ),
	pointsX( pointsX ), pointsY( pointsY )
{
	setProcessXY();
	rankX = rank % processesX;
	rankY = rank / processesX;
	GetBeginEndPoints( pointsX, processesX, rankX, beginX, endX );
	GetBeginEndPoints( pointsY, processesY, rankY, beginY, endY );

	if( hasLeftNeighbor() ) {
		beginX--;
	}
	if( hasRightNeighbor() ) {
		endX++;
	}
	if( hasTopNeighbor() ) {
		beginY--;
	}
	if( hasBottomNeighbor() ) {
		endY++;
	}

	// Инициализируем grid.
	grid.X.PartInit( area.X0, area.Xn, pointsX, beginX, endX );
	grid.Y.PartInit( area.Y0, area.Yn, pointsY, beginY, endY );

	// Заполняем список соседей с которыми будем обмениваться данными.
	setExchangeDefinitions();

	// Выполняем инициализацию.
	CBaseProgram::Initialze();

#ifdef _DEBUG
	cout << "(" << rank << ")" << " {" << rankX << ", " << rankY << "}"
		<< " [" << beginX << ", " << endX << ")"
		<< " x [" << beginY << ", " << endY << ")" << endl;
	for( CExchangeDefinitions::const_iterator i = exchangeDefinitions.begin();
		i != exchangeDefinitions.end(); ++i ) {
		cout << rankX << " " << rankY << " "
			<< i->SendPart() << " " << i->RecvPart() << endl;
	}
#endif
}

void CProgram::setProcessXY()
{
	size_t power = 0;
	{
		size_t i = 1;
		while( i < numberOfProcesses ) {
			i *= 2;
			power++;
		}
		if( i != numberOfProcesses ) {
			throw CException( "The number of processes must be power of 2." );
		}
	}

	float pX = static_cast<float>( pointsX );
	float pY = static_cast<float>( pointsY );

	size_t powerX = 0;
	size_t powerY = 0;
	for( size_t i = 0; i < power; i++ ) {
		if( pX > pY ) {
			pX = pX / 2;
			powerX++;
		} else {
			pY = pY / 2;
			powerY++;
		}
	}

	processesX = 1 << powerX;
	processesY = 1 << powerY;
}

void CProgram::setExchangeDefinitions()
{
	if( hasLeftNeighbor() ) {
		exchangeDefinitions.push_back( CExchangeDefinition(
			rankByXY( rankX - 1, rankY ),
			grid.Column( 1, 1 /* decreaseTop */, 1 /* decreaseBottom */ ),
			grid.Column( 0, 1 /* decreaseTop */, 1 /* decreaseBottom */ ) ) );
	}
	if( hasRightNeighbor() ) {
		exchangeDefinitions.push_back( CExchangeDefinition(
			rankByXY( rankX + 1, rankY ),
			grid.Column( grid.X.Size() - 2, 1 /* decreaseTop */, 1 /* decreaseBottom */ ),
			grid.Column( grid.X.Size() - 1, 1 /* decreaseTop */, 1 /* decreaseBottom */ ) ) );
	}
	if( hasTopNeighbor() ) {
		exchangeDefinitions.push_back( CExchangeDefinition(
			rankByXY( rankX, rankY - 1 ),
			grid.Row( 1, 1 /* decreaseLeft */, 1 /* decreaseRight */ ),
			grid.Row( 0, 1 /* decreaseLeft */, 1 /* decreaseRight */ ) ) );
	}
	if( hasBottomNeighbor() ) {
		exchangeDefinitions.push_back( CExchangeDefinition(
			rankByXY( rankX, rankY + 1 ),
			grid.Row( grid.Y.Size() - 2, 1 /* decreaseLeft */, 1 /* decreaseRight */ ),
			grid.Row( grid.Y.Size() - 1, 1 /* decreaseLeft */, 1 /* decreaseRight */ ) ) );
	}
}

void CProgram::allReduceFraction( CFraction& fraction )
{
	NumericType buffer[2] = { fraction.Numerator, fraction.Denominator };
	MpiCheck( MPI_Allreduce( MPI_IN_PLACE, buffer, 2 /* count */,
		MpiNumericType, MPI_SUM, MPI_COMM_WORLD ), "MPI_Allreduce" );
	fraction.Numerator = buffer[0];
	fraction.Denominator = buffer[1];
}

void CProgram::allReduceDifference()
{
	MpiCheck( MPI_Allreduce( MPI_IN_PLACE, &difference, 1 /* count */,
		MpiNumericType, MPI_MAX, MPI_COMM_WORLD ), "MPI_Allreduce" );
}

void CProgram::iteration0()
{
	p.Init( grid.X.Size(), grid.Y.Size() );

	if( !hasLeftNeighbor() ) {
		for( size_t y = 0; y < p.SizeY(); y++ ) {
			p( 0, y ) = Phi( grid.X[0], grid.Y[y] );
		}
	}
	if( !hasRightNeighbor() ) {
		const size_t left = p.SizeX() - 1;
		for( size_t y = 0; y < p.SizeY(); y++ ) {
			p( left, y ) = Phi( grid.X[left], grid.Y[y] );
		}
	}
	if( !hasTopNeighbor() ) {
		for( size_t x = 0; x < p.SizeX(); x++ ) {
			p( x, 0 ) = Phi( grid.X[x], grid.Y[0] );
		}
	}
	if( !hasBottomNeighbor() ) {
		const size_t bottom = p.SizeY() - 1;
		for( size_t x = 0; x < p.SizeX(); x++ ) {
			p( x, bottom ) = Phi( grid.X[x], grid.Y[bottom] );
		}
	}

	cudaP.Allocate( p );
}

void CProgram::iteration1()
{
	CMatrix r( grid.X.Size(), grid.Y.Size() );
	cudaR.Allocate( r );

	CalcR( cudaGridDim, cudaP, cudaGrid, cudaR );
	exchangeDefinitions.Exchange( cudaR );

	CFraction tau = CalcTau( cudaGridDim, cudaR, cudaR, cudaGrid, cudaBuffer );
	allReduceFraction( tau );

	difference = CalcP( cudaGridDim, cudaR, tau.Value(), cudaP, cudaBuffer );
	allReduceDifference();

	cudaR.Dump( r );
	cudaG.Allocate( r );
}

void CProgram::iteration2()
{
	exchangeDefinitions.Exchange( cudaP );

	CalcR( cudaGridDim, cudaP, cudaGrid, cudaR );
	exchangeDefinitions.Exchange( cudaR );

	CFraction alpha = CalcAlpha( cudaGridDim, cudaR, cudaG, cudaGrid, cudaBuffer );
	allReduceFraction( alpha );

	CalcG( cudaGridDim, cudaR, alpha.Value(), cudaG );
	exchangeDefinitions.Exchange( cudaG );

	CFraction tau = CalcTau( cudaGridDim, cudaR, cudaG, cudaGrid, cudaBuffer );
	allReduceFraction( tau );

	difference = CalcP( cudaGridDim, cudaG, tau.Value(), cudaP, cudaBuffer );
	allReduceDifference();
}

///////////////////////////////////////////////////////////////////////////////

void ParseArguments( const int argc, const char* const argv[],
	size_t& pointsX, size_t& pointsY, string& dumpFilename )
{
	if( argc < 3 || argc > 4 ) {
		throw CException( "too few arguments\n"
			"Usage: dirch POINTS_X POINTS_Y [DUMP_FILENAME]" );
	}

	pointsX = strtoul( argv[1], 0, 10 );
	pointsY = strtoul( argv[2], 0, 10 );

	if( pointsX == 0 || pointsY == 0 ) {
		throw CException( "invalid format of arguments\n"
			"Usage: dirch POINTS_X POINTS_Y [DUMP_FILENAME]" );
	}

	if( argc == 4 ) {
		dumpFilename = argv[3];
	}
}

void Main( const int argc, const char* const argv[] )
{
	double programTime = 0.0;
	{
		CMpiTimer timer( programTime );

		// Выбираем CUDA устройство.
		SelectSuitableCudaDevice( cout, CMpiSupport::Rank() );

		// Используем только потоки ввода вывода iostream,
		// поэтому отключаем синхронизацию ввода вывода со стандартной библиотекой C.
		ios::sync_with_stdio( false );

		size_t pointsX;
		size_t pointsY;
		string dumpFilename;
		ParseArguments( argc, argv, pointsX, pointsY, dumpFilename );

		auto_ptr<IIterationCallback> callback( new CSimpleIterationCallback );
		if( CMpiSupport::Rank() == 0 ) {
			callback.reset( new CIterationCallback( cout, 0 ) );
		}

		if( CMpiSupport::NumberOfProccess() == 1 ) {
			CBaseProgram::Run( pointsX, pointsY, Area, *callback, dumpFilename );
		} else {
			CProgram::Run( pointsX, pointsY, Area, *callback );
		}
	}
	cout << "(" << CMpiSupport::Rank() << ") Time: " << programTime << endl;
}

int main( int argc, char** argv )
{
	try {
		CMpiSupport::Initialize( &argc, &argv );
		Main( argc, argv );
		CMpiSupport::Finalize();
	} catch( exception& e ) {
		cerr << "Error: " << e.what() << endl;
		CMpiSupport::Abort( 1 );
		return 1;
	} catch( ... ) {
		cerr << "Unknown error!" << endl;
		CMpiSupport::Abort( 2 );
		return 2;
	}

	return 0;
}
