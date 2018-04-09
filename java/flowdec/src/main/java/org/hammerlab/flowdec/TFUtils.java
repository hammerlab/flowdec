package org.hammerlab.flowdec;

import org.tensorflow.Graph;
import org.tensorflow.framework.GraphDef;

import com.google.protobuf.InvalidProtocolBufferException;

public class TFUtils {

	/**
	 * Set device associated with TensorFlow graph execution
	 * 
	 * @param g TensorFlow graph instance
	 * @param device Name of device to place execution on; e.g. "/cpu:0", "/gpu:1", etc.
	 * See <a href="https://www.tensorflow.org/programmers_guide/faq">TF FAQ</a> for more details.
	 * @return Graph with all operations assigned to given device
	 */
	public static Graph setDevice(Graph g, String device) {
		
		// This functionality is based entirely on:
		// https://stackoverflow.com/questions/47799972/tensorflow-java-multi-gpu-inference

		GraphDef.Builder builder;
		try {
			builder = GraphDef.parseFrom(g.toGraphDef()).toBuilder();
		} catch (InvalidProtocolBufferException e) {
			throw new RuntimeException(e);
		}
		
		for (int i = 0; i < builder.getNodeCount(); ++i) {
			builder.getNodeBuilder(i).setDevice(device);
		}
		
		Graph gd = new Graph();
		gd.importGraphDef(builder.build().toByteArray());
		return gd;
	}
	
	
}
