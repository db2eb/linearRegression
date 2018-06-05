//Data input
const x_vals = [],
	y_vals = [],
//tf.js sgd 
	learningRate = 0.5,
	optimizer = tf.train.sgd(learningRate);
//Linear Regression: y = mx+b
let m,b;

function setup(){
	var canv = createCanvas(400,400);
	canv.mousePressed(addDot);
	m = tf.variable(tf.scalar(1));
	b = tf.variable(tf.scalar(0));
}

//Draw points
function addDot(){
	let x = map(mouseX,0,width,0,1), //Normalize
		y = map(mouseY,0,height,1,0);
	x_vals.push(x);
	y_vals.push(y);
	console.log("yes");
}

function loss(pred, labels){
	return pred.sub(labels).square().mean();
}

function predict(x){
	const xs = tf.tensor1d(x),
		  ys = xs.mul(m).add(b);// y = mx+b;
	return ys; // tensor
}

function draw(){
	tf.tidy(()=>{
		if(x_vals.length>0){
			const ys = tf.tensor1d(y_vals); // Convert array into tensor
			optimizer.minimize(() => loss(predict(x_vals), ys));
		}
	});

	background(250);
	stroke(0);
	strokeWeight(8);

	for(let i=0;i<x_vals.length;i++){
		let px = map(x_vals[i],0,1,0,width),
			py = map(y_vals[i],0,1,height,0);
		point(px,py);
	}

	const lineX = [0,1],
		  ys = tf.tidy(()=> predict(lineX));
	let lineY = ys.dataSync();
	ys.dispose();

	let x1 = map(lineX[0],0,1,0,width),
		x2 = map(lineX[1],0,1,0,width),
		y1 = map(lineY[0],0,1,height,0),
		y2 = map(lineY[1],0,1,height,0);
	strokeWeight(2);
	line(x1,y1,x2,y2);
}
