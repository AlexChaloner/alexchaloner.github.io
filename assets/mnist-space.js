(function () {
  "use strict";
  const mount = document.getElementById("mnist-pca");
  if (!mount) return;
  const colors = ["#2679b2", "#dc7436", "#31885a", "#cc5054", "#8c62b1", "#956444", "#c75b9d", "#394a73", "#9b8a22", "#23989f"];
  const ready = fetch(mount.dataset.pointsUrl).then(response => {
    if (!response.ok) throw new Error("Could not load the fixed MNIST space");
    return response.json();
  });

  function frame(ctx, width, height, data) {
    const xs = data.points.map(p => p[0]), ys = data.points.map(p => p[1]);
    const bounds = data.bounds || [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const left = 58, right = width - 22, top = 22, bottom = height - 48;
    const scale = Math.min((right-left)/((bounds[1]-bounds[0])*1.12), (bottom-top)/((bounds[3]-bounds[2])*1.12));
    const midX = (bounds[0]+bounds[1])/2, midY = (bounds[2]+bounds[3])/2;
    const px = x => (left+right)/2+(x-midX)*scale;
    const py = y => (top+bottom)/2-(y-midY)*scale;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#fbfcf8"; ctx.fillRect(0, 0, width, height);
    ctx.font = "11px system-ui, sans-serif";
    const rawStep = Math.max(bounds[1]-bounds[0], bounds[3]-bounds[2])/6;
    const magnitude = 10 ** Math.floor(Math.log10(rawStep));
    const step = [1,2,5,10].find(v => v*magnitude >= rawStep)*magnitude;
    const minimum = Math.min(midX-(right-left)/(2*scale),midY-(bottom-top)/(2*scale));
    const maximum = Math.max(midX+(right-left)/(2*scale),midY+(bottom-top)/(2*scale));
    for (let tick=Math.ceil(minimum/step);tick<=Math.floor(maximum/step);tick++) {
      const value=Number((tick*step).toPrecision(6)), x=px(value), y=py(value);
      ctx.strokeStyle=value===0 ? "#b9c5be" : "#e4e9e2"; ctx.fillStyle="#5f6d66";
      if(x>=left && x<=right) {
        ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,bottom);ctx.stroke();
        ctx.textAlign="center";ctx.fillText(value,x,bottom+17);
      }
      if(y>=top && y<=bottom) {
        ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();
        ctx.textAlign="right";ctx.fillText(value,left-9,y+4);
      }
    }
    ctx.fillStyle="#5f6d66";ctx.textAlign="center";
    ctx.fillText(`PC1 · ${(100*data.explainedVariance[0]).toFixed(1)}%`,(left+right)/2,height-10);
    ctx.save();ctx.translate(15,(top+bottom)/2);ctx.rotate(-Math.PI/2);
    ctx.fillText(`PC2 · ${(100*data.explainedVariance[1]).toFixed(1)}%`,0,0);ctx.restore();
    return point => [px(point[0]),py(point[1])];
  }

  function reference(ctx, data, screen, opacity=.2) {
    ctx.save();ctx.globalAlpha=opacity;
    data.points.forEach((point,i)=>{
      const [x,y]=screen(point);
      ctx.fillStyle=data.labels[i]===null ? "#888888" : colors[data.labels[i]];
      ctx.beginPath();ctx.arc(x,y,2,0,2*Math.PI);ctx.fill();
    });
    ctx.restore();
  }

  window.MNISTSpace = {ready, frame, reference, colors};
}());
