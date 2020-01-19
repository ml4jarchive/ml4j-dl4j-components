package org.ml4j.nn.components.activationfunctions;

import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.components.activationfunctions.base.DifferentiableActivationFunctionComponentActivationTestBase;
import org.ml4j.nn.components.mocks.MockTestData;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;
import org.mockito.Mock;
import org.mockito.Mockito;

public class DL4JDifferentiableActivationFunctionComponentActivationImplTest extends DifferentiableActivationFunctionComponentActivationTestBase<DifferentiableActivationFunctionComponentAdapter> {

	@Mock
	private DifferentiableActivationFunctionActivation mockActivationFunctionActivation;
	
	@Mock
	private NeuronsActivationContext mockActivationContext;
	
	// TODO THUR
	@Mock
	protected DifferentiableActivationFunction mockActivationFunction;
	
	@Mock
	protected DifferentiableActivationFunctionComponentAdapter mockDifferentiableActivationFunctionComponentAdapter;
	
	@Override
	protected DifferentiableActivationFunctionComponentActivation createDifferentiableActivationFunctionComponentActivationUnderTest(
			DifferentiableActivationFunctionComponentAdapter activationFunctionComponent, NeuronsActivation input, NeuronsActivation output) {
		
	    Mockito.when(mockDifferentiableActivationFunctionComponentAdapter.getActivationFunction()).thenReturn(mockActivationFunction);

		
		Mockito.when(mockActivationFunctionActivation.getActivationFunction()).thenReturn(mockActivationFunction);
		Mockito.when(mockActivationFunctionActivation.getInput()).thenReturn(input);
		Mockito.when(mockActivationFunctionActivation.getOutput()).thenReturn(output);
	
		return new DefaultDifferentiableActivationFunctionComponentActivationImpl(activationFunctionComponent, mockActivationFunctionActivation, mockActivationContext);
	}

	@Override
	public void testBackPropagate() {
		
		NeuronsActivation mockActivationFunctionGradient = createNeuronsActivation(mockOutputActivation.getFeatureCount(), mockOutputActivation.getExampleCount());
		
		Mockito.when(mockActivationFunction.activationGradient(mockActivationFunctionActivation, mockActivationContext)).thenReturn(mockActivationFunctionGradient);
		
		super.testBackPropagate();
	}

	@Override
	protected MatrixFactory createMatrixFactory() {
		return new JBlasRowMajorMatrixFactory();
	}

	@Override
	public NeuronsActivation createNeuronsActivation(int featureCount, int exampleCount) {
		return MockTestData.mockNeuronsActivation(featureCount, exampleCount, matrixFactory);
	}

	@Override
	protected DifferentiableActivationFunctionComponentAdapter createMockActivationFunctionComponent() {
		return mockDifferentiableActivationFunctionComponentAdapter;
	}
}
