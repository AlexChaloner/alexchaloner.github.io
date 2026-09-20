(async function () {
  "use strict";
  const mount=document.getElementById("sampling-paths");
  if(!mount)return;
  // Load the larger recording only when the reader approaches this experiment.
  if("IntersectionObserver" in window)await new Promise(resolve=>{
    const observer=new IntersectionObserver(entries=>{if(entries.some(entry=>entry.isIntersecting)){observer.disconnect();resolve();}},{rootMargin:"800px"});
    observer.observe(mount);
  });
  const role=name=>mount.querySelector(`[data-role="${name}"]`);
  const control=name=>mount.querySelector(`[data-control="${name}"]`);
  try {
    const atlas=new Image();atlas.src=mount.dataset.atlasUrl;
    const [reference,data]=await Promise.all([window.MNISTSpace.ready,
      fetch(mount.dataset.routesUrl).then(response=>{if(!response.ok)throw Error("Could not load sampling paths");return response.json();}),atlas.decode()]);
    const space=window.MNISTSpace;
    if(data.projectionId!==reference.projectionId)throw Error("Sampling paths use different PCA axes");
    const charts=[...mount.querySelectorAll("canvas[data-method]")];
    let run=data.runs.find(r=>r.steps===data.defaultSteps),sampler="ddpm",example=0,step=run.steps-1;
    let playing=false,paused=false,animation=0,viewBounds=null,selection=null;
    const screens=new Map(),positions=new Map(),hover=new Map(),tooltips=new Map(),dragged=new Set();
    const methodFor=canvas=>canvas.dataset.method==="diffusion"?sampler:"flow";
    data.runs.forEach(r=>control("steps").add(new Option(r.steps.toLocaleString(),r.steps)));
    control("steps").value=String(run.steps);
    data.exampleNames.forEach((name,i)=>control("example").add(new Option(name,i)));
    charts.forEach(canvas=>{
      const tip=document.createElement("div");tip.className="mnist-route-tooltip";tip.hidden=true;tip.setAttribute("role","tooltip");
      canvas.parentElement.append(tip);tooltips.set(canvas,tip);
    });
    function tile(canvas,id) {
      canvas.getContext("2d").drawImage(atlas,id%data.atlasColumns*data.imageSide,Math.floor(id/data.atlasColumns)*data.imageSide,
        data.imageSide,data.imageSide,0,0,data.imageSide,data.imageSide);
    }
    function clearHover(){hover.clear();tooltips.forEach(tip=>{tip.hidden=true;});}
    function playbackControls(){control("play").textContent=playing||paused?"Restart route":step===run.steps-1?"Replay route":"Play route";control("pause").disabled=!playing&&!paused;control("pause").textContent=paused?"Resume":"Pause";}
    function stop(){cancelAnimationFrame(animation);playing=false;paused=false;playbackControls();}
    function draw(canvas) {
      const {width,height}=canvas.getBoundingClientRect(),dpr=Math.min(2,devicePixelRatio||1);
      canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);
      const ctx=canvas.getContext("2d");ctx.setTransform(dpr,0,0,dpr,0,0);
      const screen=space.frame(ctx,width,height,reference,viewBounds),rect=screen.plotRect;
      screens.set(canvas,screen);
      ctx.save();ctx.beginPath();ctx.rect(rect.left,rect.top,rect.right-rect.left,rect.bottom-rect.top);ctx.clip();
      space.reference(ctx,reference,screen,.14);
      const paths=run.methods[methodFor(canvas)].map(ids=>ids.map(id=>screen(data.points[id])));
      positions.set(canvas,paths);
      const color=canvas.dataset.method==="diffusion"?"#7857b2":"#167d69";
      const order=paths.map((_,i)=>i).filter(i=>i!==example).concat(example);
      const line=(points,end)=>{ctx.beginPath();ctx.moveTo(...points[0]);for(let i=1;i<=end;i++)ctx.lineTo(...points[i]);ctx.stroke();};
      order.forEach(lane=>{
        const points=paths[lane],chosen=lane===example,highlighted=hover.get(canvas)?.lane===lane;
        ctx.strokeStyle=color;ctx.lineWidth=chosen||highlighted?1.6:1;
        ctx.globalAlpha=chosen ? .25 : .10;line(points,run.steps);
        ctx.globalAlpha=chosen||highlighted ? .9 : .22;line(points,step+1);
        const [x,y]=points[step];ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,y,chosen?4:2,0,Math.PI*2);ctx.fill();
        if(chosen){const [nx,ny]=points[step+1],dx=nx-x,dy=ny-y,length=Math.hypot(dx,dy);
          if(length>.1){const size=Math.min(6,length*.5),ux=dx/length,uy=dy/length;ctx.globalAlpha=1;ctx.beginPath();ctx.moveTo(nx,ny);ctx.lineTo(nx-size*ux+size*.45*uy,ny-size*uy-size*.45*ux);ctx.lineTo(nx-size*ux-size*.45*uy,ny-size*uy+size*.45*ux);ctx.closePath();ctx.fill();}}
      });
      ctx.restore();
      if(viewBounds){ctx.save();ctx.strokeStyle="#5f6d66";ctx.setLineDash([4,3]);ctx.strokeRect(rect.left,rect.top,rect.right-rect.left,rect.bottom-rect.top);ctx.restore();}
      if(selection?.canvas===canvas){ctx.save();ctx.fillStyle="rgba(38,121,178,.12)";ctx.strokeStyle="#2679b2";ctx.setLineDash([4,3]);ctx.fillRect(selection.x,selection.y,selection.endX-selection.x,selection.endY-selection.y);ctx.strokeRect(selection.x,selection.y,selection.endX-selection.x,selection.endY-selection.y);ctx.restore();}
      const point=data.points[run.methods[methodFor(canvas)][example][step]];
      canvas.setAttribute("aria-label",`${canvas.dataset.method==="diffusion"?`Diffusion, ${sampler==="ddpm"?"with":"without"} added sampling noise`:"Flow matching, Euler"}. ${data.exampleNames[example]}. Step ${step} to ${step+1} of ${run.steps}. PC1 ${point[0].toFixed(2)}, PC2 ${point[1].toFixed(2)}. Fixed PC1/PC2 axes.${viewBounds?` Zoomed to PC1 ${viewBounds[0]}–${viewBounds[1]}, PC2 ${viewBounds[2]}–${viewBounds[3]}.`:""} Drag to zoom; double-click or Escape to reset. Arrow keys change steps and examples; Home and End jump to the first and last transition.`);
    }
    function update() {
      charts.forEach(draw);
      control("step").max=String(run.steps-1);control("step").value=String(step);control("example").value=String(example);
      role("step").textContent=`${step} → ${step+1} / ${run.steps.toLocaleString()}`;
      role("diffusion-sampler").textContent=sampler==="ddpm"?"DDPM · added noise":"DDIM · no added noise";
      for(const method of ["diffusion","flow"]){const path=run.methods[method==="diffusion"?sampler:method][example];
        for(const [name,index] of [["start",0],["now",step],["next",step+1]]){
          const canvas=mount.querySelector(`[data-image="${method}-${name}"]`);tile(canvas,path[index]);
          canvas.setAttribute("aria-label",`${method} ${name}: ${data.exampleNames[example]}, step ${index}`);
        }
      }
      control("reset-zoom").disabled=!viewBounds;playbackControls();
    }
    function play(restart){stop();clearHover();if(restart||step===run.steps-1)step=0;playing=true;const start=performance.now(),initial=step;
      function frame(now){if(!playing)return;const next=Math.min(run.steps-1,initial+Math.floor((now-start)*run.steps/10000));if(next!==step){step=next;update();}if(step===run.steps-1){stop();return;}animation=requestAnimationFrame(frame);}
      update();animation=requestAnimationFrame(frame);
    }
    function cancelSelection(){if(!selection)return;const {canvas,pointerId}=selection;selection=null;if(canvas.hasPointerCapture(pointerId))canvas.releasePointerCapture(pointerId);}
    function resetZoom(){cancelSelection();clearHover();viewBounds=null;update();}
    control("steps").addEventListener("change",()=>{stop();cancelSelection();clearHover();const progress=step/(run.steps-1);run=data.runs.find(r=>r.steps===Number(control("steps").value));step=Math.round(progress*(run.steps-1));update();});
    control("sampler").addEventListener("change",()=>{stop();clearHover();sampler=control("sampler").value;update();});
    control("example").addEventListener("change",()=>{stop();clearHover();example=Number(control("example").value);update();});
    control("step").addEventListener("input",()=>{stop();clearHover();step=Number(control("step").value);update();});
    control("play").addEventListener("click",()=>play(true));
    control("pause").addEventListener("click",()=>{if(paused){play(false);return;}cancelAnimationFrame(animation);playing=false;paused=true;playbackControls();});
    control("reset-zoom").addEventListener("click",resetZoom);
    function nearest(canvas,event){const box=canvas.getBoundingClientRect(),x=event.clientX-box.left,y=event.clientY-box.top,rect=screens.get(canvas).plotRect;
      if(x<rect.left||x>rect.right||y<rect.top||y>rect.bottom)return null;
      let found=null,distance=(event.pointerType==="touch"?20:8)**2;
      positions.get(canvas).forEach((points,lane)=>{for(let i=0;i<run.steps;i++){const [ax,ay]=points[i],[bx,by]=points[i+1],dx=bx-ax,dy=by-ay,t=Math.max(0,Math.min(1,((x-ax)*dx+(y-ay)*dy)/(dx*dx+dy*dy||1))),d=(x-ax-t*dx)**2+(y-ay-t*dy)**2;
        if(d<distance){distance=d;found={lane,step:i,x,y};}}});return found;
    }
    function moveSelection(event){const box=selection.canvas.getBoundingClientRect(),{rect,x,y}=selection;selection.endX=Math.max(rect.left,Math.min(rect.right,event.clientX-box.left));selection.endY=Math.max(rect.top,Math.min(rect.bottom,event.clientY-box.top));if(Math.hypot(selection.endX-x,selection.endY-y)>=8)dragged.add(selection.canvas);}
    charts.forEach(canvas=>{
      canvas.addEventListener("pointerdown",event=>{if(event.button!==0||!event.isPrimary||selection)return;dragged.delete(canvas);const box=canvas.getBoundingClientRect(),x=event.clientX-box.left,y=event.clientY-box.top,screen=screens.get(canvas),rect=screen.plotRect;if(x<rect.left||x>rect.right||y<rect.top||y>rect.bottom)return;
        stop();clearHover();canvas.focus({preventScroll:true});const origin=screen([0,0]),unit=screen([1,1]);selection={canvas,pointerId:event.pointerId,x,y,endX:x,endY:y,rect,unproject:(a,b)=>[(a-origin[0])/(unit[0]-origin[0]),(b-origin[1])/(unit[1]-origin[1])]};canvas.setPointerCapture(event.pointerId);
      });
      canvas.addEventListener("pointermove",event=>{if(selection){if(selection.canvas===canvas&&selection.pointerId===event.pointerId){moveSelection(event);draw(canvas);}return;}if(event.pointerType==="touch")return;
        const hit=nearest(canvas,event),before=hover.get(canvas),tip=tooltips.get(canvas);if(hit){hover.set(canvas,hit);tip.textContent=`${data.exampleNames[hit.lane]} · ${hit.step} → ${hit.step+1}`;tip.hidden=false;tip.style.left=`${Math.max(4,Math.min(canvas.clientWidth-tip.offsetWidth-4,hit.x+12))}px`;tip.style.top=`${Math.max(canvas.offsetTop+4,canvas.offsetTop+hit.y-tip.offsetHeight-8)}px`;}else{hover.delete(canvas);tip.hidden=true;}if(before?.lane!==hit?.lane)draw(canvas);
      });
      canvas.addEventListener("pointerup",event=>{if(selection?.canvas!==canvas||selection.pointerId!==event.pointerId)return;moveSelection(event);const {x,y,endX,endY,unproject}=selection;
        if(Math.abs(endX-x)>=8&&Math.abs(endY-y)>=8){const [left,top]=unproject(Math.min(x,endX),Math.min(y,endY)),[right,bottom]=unproject(Math.max(x,endX),Math.max(y,endY));if(right-left>=.001&&top-bottom>=.001)viewBounds=[left,right,bottom,top];}
        cancelSelection();update();
      });
      ["pointercancel","lostpointercapture"].forEach(type=>canvas.addEventListener(type,()=>{if(selection?.canvas!==canvas)return;dragged.add(canvas);cancelSelection();update();}));
      canvas.addEventListener("pointerleave",()=>{hover.delete(canvas);tooltips.get(canvas).hidden=true;draw(canvas);});
      canvas.addEventListener("click",event=>{if(dragged.delete(canvas))return;const hit=nearest(canvas,event);if(hit){stop();clearHover();example=hit.lane;step=hit.step;update();}});
      canvas.addEventListener("dblclick",event=>{event.preventDefault();resetZoom();});
      canvas.addEventListener("keydown",event=>{if(!["ArrowLeft","ArrowRight","ArrowUp","ArrowDown","Home","End","Escape"].includes(event.key))return;event.preventDefault();stop();clearHover();
        if(event.key==="Escape"){if(selection){dragged.add(canvas);cancelSelection();update();}else resetZoom();return;}
        if(event.key==="ArrowUp"||event.key==="ArrowDown")example=(example+(event.key==="ArrowDown"?1:data.exampleNames.length-1))%data.exampleNames.length;
        else step=event.key==="Home"?0:event.key==="End"?run.steps-1:Math.max(0,Math.min(run.steps-1,step+(event.key==="ArrowRight"?1:-1)));update();
      });
    });
    role("content").hidden=false;role("status").hidden=true;update();
    const observer=new ResizeObserver(()=>{cancelSelection();clearHover();charts.forEach(draw);});charts.forEach(canvas=>observer.observe(canvas));
  } catch(error){role("status").textContent="The sampling paths couldn’t load. Refresh to try again.";console.error(error);}
}());
