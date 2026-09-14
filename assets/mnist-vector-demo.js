(async function () {
  "use strict";
  const mount=document.querySelector(".mnist-vector-demo"), space=window.MNISTSpace;
  if(!mount || !space)return;
  const control=name=>mount.querySelector(`[data-vector-control="${name}"]`);
  const role=name=>mount.querySelector(`[data-vector-role="${name}"]`);
  const status=mount.querySelector(".mnist-vector-status");
  try {
    const atlas=new Image();atlas.src=mount.dataset.atlasUrl;
    const [reference,data]=await Promise.all([space.ready,
      fetch(mount.dataset.routesUrl).then(response=>{if(!response.ok)throw Error("Could not load image vectors");return response.json();}),atlas.decode()]);
    if(data.projectionId!==reference.projectionId)throw Error("Vector data does not match the fixed PCA axes");
    const snapshots=data.snapshots.filter(snapshot=>snapshot.update>0), names=data.exampleNames;
    let checkpoint=Math.max(0,snapshots.findIndex(snapshot=>snapshot.update===100)), example=0, step=12;
    const charts=[...mount.querySelectorAll("canvas[data-vector-method]")], arrows=new Map(), hovered=new Map(), tooltips=new Map();
    snapshots.forEach((snapshot,i)=>control("checkpoint").add(new Option(`${snapshot.update.toLocaleString()} updates`,i)));
    control("checkpoint").value=String(checkpoint);
    names.forEach((name,i)=>control("example").add(new Option(name,i)));
    control("step").max=String(data.solverSteps-1);
    charts.forEach(canvas=>{
      const tooltip=document.createElement("div");tooltip.className="mnist-route-tooltip mnist-vector-tooltip";tooltip.hidden=true;
      canvas.parentElement.append(tooltip);tooltips.set(canvas,tooltip);
    });
    function tile(ctx,id,x,y,size) {
      ctx.drawImage(atlas,(id%data.atlasColumns)*data.imageSide,Math.floor(id/data.atlasColumns)*data.imageSide,
        data.imageSide,data.imageSide,x,y,size,size);
    }
    function imageCard(label,id) {
      const figure=document.createElement("figure"), caption=document.createElement("figcaption"), canvas=document.createElement("canvas");
      caption.textContent=label;canvas.width=canvas.height=data.imageSide;
      canvas.setAttribute("role","img");canvas.setAttribute("aria-label",label);tile(canvas.getContext("2d"),id,0,0,data.imageSide);
      figure.append(caption,canvas);return figure;
    }
    function inspector(method) {
      const path=snapshots[checkpoint][method][example], from=data.points[path[step]], to=data.points[path[step+1]];
      const box=document.createElement("article");box.className="mnist-route-inspector";
      const title=document.createElement("h4");title.textContent=`${method==="diffusion"?"Diffusion":"Flow matching"} · ${names[example]}`;
      const delta=document.createElement("p");delta.className="mnist-route-coordinates";
      const signed=value=>`${value>=0?"+":""}${value.toFixed(2)}`;
      delta.textContent=`ΔPC1 ${signed(to[0]-from[0])} · ΔPC2 ${signed(to[1]-from[1])}`;
      const cards=document.createElement("div");cards.className="mnist-route-cards";
      cards.append(imageCard(`From · step ${step}`,path[step]),imageCard(`To · step ${step+1}`,path[step+1]));
      const copy=document.createElement("p");copy.className="mnist-route-note";
      copy.textContent=method==="diffusion"?"The predicted noise is used to estimate the clean image and compute the next sampling state.":"The predicted pixel velocity is multiplied by the step size and added to the current image.";
      box.append(title,delta,cards,copy);return box;
    }
    function draw(canvas) {
      const {width,height}=canvas.getBoundingClientRect(), ratio=Math.min(2,window.devicePixelRatio||1);
      canvas.width=Math.max(1,Math.round(width*ratio));canvas.height=Math.max(1,Math.round(height*ratio));
      const ctx=canvas.getContext("2d");ctx.setTransform(ratio,0,0,ratio,0,0);
      const screen=space.frame(ctx,width,height,reference);space.reference(ctx,reference,screen,.18);
      const method=canvas.dataset.vectorMethod, paths=snapshots[checkpoint][method], color=method==="diffusion"?"#7857b2":"#167d69";
      const items=[];
      if(control("context").checked)for(let t=0;t<data.solverSteps;t+=3)if(t!==step)paths.forEach((path,lane)=>items.push({lane,step:t}));
      paths.forEach((path,lane)=>{if(lane!==example)items.push({lane,step});});items.push({lane:example,step});
      items.forEach(item=>{
        const path=paths[item.lane];item.from=screen(data.points[path[item.step]]);item.to=screen(data.points[path[item.step+1]]);
        const active=item.step===step, selected=active&&item.lane===example, hover=hovered.get(canvas);
        const highlighted=hover?.lane===item.lane&&hover?.step===item.step;
        ctx.globalAlpha=selected||highlighted ? 1 : active ? .7 : .22;ctx.strokeStyle=color;ctx.fillStyle=color;
        ctx.lineWidth=selected||highlighted?2.5:active?1.8:1;
        const [x,y]=item.from, dx=item.to[0]-x, dy=item.to[1]-y, length=Math.hypot(dx,dy);
        ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(...item.to);ctx.stroke();
        if(length>.1) {
          const head=Math.min(6,length*.45), ux=dx/length, uy=dy/length, [endX,endY]=item.to;
          ctx.beginPath();ctx.moveTo(endX,endY);ctx.lineTo(endX-head*ux+head*.45*uy,endY-head*uy-head*.45*ux);
          ctx.lineTo(endX-head*ux-head*.45*uy,endY-head*uy+head*.45*ux);ctx.closePath();ctx.fill();
        }
        ctx.beginPath();ctx.arc(x,y,selected?3:active?2:1.2,0,2*Math.PI);ctx.fill();
      });
      ctx.globalAlpha=1;arrows.set(canvas,items);
      const path=paths[example], from=data.points[path[step]], to=data.points[path[step+1]];
      canvas.setAttribute("aria-label",`${method==="diffusion"?"Diffusion":"Flow matching"}, ${names[example]}, step ${step} to ${step+1}. From PC1 ${from[0].toFixed(2)}, PC2 ${from[1].toFixed(2)} to PC1 ${to[0].toFixed(2)}, PC2 ${to[1].toFixed(2)}. Left/right changes step; up/down changes example.`);
    }
    function render() {
      control("step").value=String(step);control("example").value=String(example);
      role("step").textContent=`${step} → ${step+1} / ${data.solverSteps}`;
      charts.forEach(draw);role("inspectors").replaceChildren(inspector("diffusion"),inspector("flow"));
    }
    function clearHover() {hovered.clear();tooltips.forEach(tooltip=>{tooltip.hidden=true;});}
    function nearest(canvas,event) {
      const rect=canvas.getBoundingClientRect(), x=event.clientX-rect.left, y=event.clientY-rect.top;
      const touch=event.pointerType==="touch"||matchMedia("(pointer: coarse)").matches;
      let best=null, distance=(touch?18:9)**2;
      for(const item of arrows.get(canvas)||[]) {
        const [ax,ay]=item.from, dx=item.to[0]-ax, dy=item.to[1]-ay;
        const t=Math.max(0,Math.min(1,((x-ax)*dx+(y-ay)*dy)/(dx*dx+dy*dy||1)));
        const d=(x-ax-t*dx)**2+(y-ay-t*dy)**2;
        if(d<=distance){best={...item,x,y};distance=d;}
      }
      return best;
    }
    for(const name of ["checkpoint","example","context"])control(name).addEventListener("change",()=>{
      clearHover();checkpoint=Number(control("checkpoint").value);example=Number(control("example").value);render();
    });
    control("step").addEventListener("input",()=>{clearHover();step=Number(control("step").value);render();});
    charts.forEach(canvas=>{
      canvas.addEventListener("pointermove",event=>{
        if(event.pointerType==="touch")return;
        const item=nearest(canvas,event), tooltip=tooltips.get(canvas);
        if(item) {
          hovered.set(canvas,item);const path=snapshots[checkpoint][canvas.dataset.vectorMethod][item.lane];
          const title=document.createElement("strong");title.textContent=`${names[item.lane]} · ${item.step} → ${item.step+1}`;
          const preview=document.createElement("canvas");preview.width=68;preview.height=32;preview.setAttribute("aria-hidden","true");
          const ctx=preview.getContext("2d");tile(ctx,path[item.step],0,0,32);tile(ctx,path[item.step+1],36,0,32);
          tooltip.replaceChildren(title,preview);tooltip.hidden=false;
          tooltip.style.left=`${Math.max(4,Math.min(canvas.clientWidth-tooltip.offsetWidth-4,item.x+12))}px`;
          tooltip.style.top=`${Math.max(canvas.offsetTop+4,canvas.offsetTop+item.y-tooltip.offsetHeight-12)}px`;
        }else{hovered.delete(canvas);tooltip.hidden=true;}
        canvas.style.cursor=item?"pointer":"crosshair";draw(canvas);
      });
      canvas.addEventListener("pointerleave",()=>{hovered.delete(canvas);tooltips.get(canvas).hidden=true;draw(canvas);});
      canvas.addEventListener("click",event=>{const item=nearest(canvas,event);if(item){example=item.lane;step=item.step;clearHover();render();}});
      canvas.addEventListener("keydown",event=>{
        if(!["ArrowLeft","ArrowRight","ArrowUp","ArrowDown","Home","End","Escape"].includes(event.key))return;
        event.preventDefault();clearHover();
        if(event.key==="ArrowLeft")step=Math.max(0,step-1);
        if(event.key==="ArrowRight")step=Math.min(data.solverSteps-1,step+1);
        if(event.key==="Home")step=0;
        if(event.key==="End")step=data.solverSteps-1;
        if(event.key==="ArrowUp")example=(example+names.length-1)%names.length;
        if(event.key==="ArrowDown")example=(example+1)%names.length;
        render();
      });
    });
    mount.querySelector(".mnist-vector-content").hidden=false;
    status.textContent="Recorded image steps · fixed PC1/PC2";
    const observer=new ResizeObserver(()=>{clearHover();charts.forEach(draw);});charts.forEach(canvas=>observer.observe(canvas));
    render();
  }catch(error){status.textContent="The image vectors couldn’t load. Refresh the page to try again.";console.error(error);}
}());
