package org.hammerlab.tfdecon;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.Callable;

import org.hammerlab.tfdecon.TFDeconResults.TFDeconResult;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Session.Runner;
import org.tensorflow.Tensor;
import org.tensorflow.framework.ConfigProto;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SignatureDef;

import com.google.protobuf.InvalidProtocolBufferException;

public class TFDeconProcessor implements Callable<TFDeconResult>, Runnable {

	private final TFDeconTask task;
	private TFDeconResult result;
	
	// See: https://github.com/tensorflow/tensorflow/blob/r1.6/tensorflow/python/saved_model/signature_constants.py
	private static final String DEFAULT_SERVING_SIGNATURE_DEF_KEY = "serving_default";
	
	TFDeconProcessor(TFDeconTask task) {
		this.task = task;
	}
	
	private MetaGraphDef getMetaGraph() {
		try {
			return MetaGraphDef.parseFrom(this.task.builder.model.metaGraphDef());
		} catch (InvalidProtocolBufferException e) {
			throw new RuntimeException("Failed to retrieve meta graph from "
					+ "saved model at '" + this.task.builder.path + "'", e);
		}
	}
	
	
	@Override
	public synchronized void run() {

		if (this.result != null) {
			return;
		}
		
		Graph g = this.task.builder.model.graph();
		if (this.task.builder.device.isPresent()) {
			g = TFUtils.setDevice(g, this.task.builder.device.get());	
		}
		
		MetaGraphDef mg = this.getMetaGraph();
		SignatureDef sd = mg.getSignatureDefMap().get(DEFAULT_SERVING_SIGNATURE_DEF_KEY);
		
		Optional<ConfigProto> conf = this.task.builder.sessionConfig;
		try (Session sess = conf.isPresent() ? 
				new Session(g, conf.get().toByteArray()) : new Session(g)){
			
			Runner runner = sess.runner();
			
			Set<String> inputNames = sd.getInputsMap().keySet();
			Set<String> outputNames = sd.getOutputsMap().keySet();
			for (String name : this.task.builder.inputs.keySet()) {
				// Get name of tensor corresponding to input name; this is typically
				// similar to the name of the input itself but does need to be (eg
				// the input name may be "data" while the tensor placeholder is "data:0")
				if (!inputNames.contains(name)) {
					throw new IllegalArgumentException("Graph input '" + 
							name + "' not valid (valid inputs = '" + inputNames + ")");
				}
				String inputTensorName = sd.getInputsMap().get(name).getName();
				runner.feed(inputTensorName, this.task.builder.inputs.get(name));
			}
			for (String name : this.task.builder.outputs) {
				if (!outputNames.contains(name)) {
					throw new IllegalArgumentException("Graph output '" + 
							name + "' not valid (valid outputs = '" + outputNames + ")");
				}
				String outputTensorName = sd.getOutputsMap().get(name).getName();
				runner.fetch(outputTensorName);
			}

			Map<String, Tensor<?>> map = new LinkedHashMap<>();
			List<Tensor<?>> tensors = runner.run();
			for (int i = 0; i < this.task.builder.outputs.size(); i++) {
				map.put(this.task.builder.outputs.get(i), tensors.get(i));
			}
			this.result = new TFDeconResult(map);
		}
	}

	public synchronized TFDeconResult getResult() {
		return this.result;
	}
	
	@Override
	public synchronized TFDeconResult call() {
		this.run();
		return this.getResult();
	}
	
	
}
