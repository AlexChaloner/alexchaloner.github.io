(async function () {
  "use strict";
  const space = window.MNISTSpace;
  if (!space) return;
  const atlases = new Map();
  function loadAtlas(url) {
    if (!atlases.has(url)) {
      const image = new Image(); image.src = url;
      atlases.set(url, image.decode().then(() => image));
    }
    return atlases.get(url);
  }
  function imageTile(ctx, atlas, id, x, y, size, data) {
    ctx.drawImage(atlas, (id % data.atlasColumns)*data.imageSide, Math.floor(id/data.atlasColumns)*data.imageSide,
      data.imageSide, data.imageSide, x, y, size, size);
  }
  function signalCanvas(values, scale) {
    if(!Array.isArray(values)) {
      const raw=atob(values.bytes), maximum=values.scale;
      values=Array.from(raw,c=>{const v=c.charCodeAt(0);return (v>127?v-256:v)*maximum/127;});
    }
    const side=Math.sqrt(values.length);
    const canvas = document.createElement("canvas"); canvas.width = canvas.height = side;
    canvas.setAttribute("role", "img"); canvas.setAttribute("aria-label", "Signed pixel signal: blue positive, orange negative");
    const ctx = canvas.getContext("2d"), pixels = ctx.createImageData(side,side);
    const maximum = scale || Math.max(.001,...values.map(Math.abs));
    values.forEach((v,i) => {
      const intensity = Math.min(1,Math.abs(v)/maximum), color = v>=0 ? [70,155,230] : [239,147,62];
      for (let c=0;c<3;c++) pixels.data[4*i+c] = Math.round(20+(color[c]-20)*intensity);
      pixels.data[4*i+3] = 255;
    });
    ctx.putImageData(pixels,0,0); return canvas;
  }
  await Promise.all([...document.querySelectorAll(".mnist-route-demo")].map(async mount => {
    const role = name => mount.querySelector(`[data-role="${name}"]`);
    const control = name => mount.querySelector(`[data-control="${name}"]`);
    const status = mount.querySelector(".mnist-route-status");
    try {
      const [reference, data, atlas] = await Promise.all([space.ready,
        fetch(mount.dataset.routesUrl).then(response => {if(!response.ok) throw Error("Could not load recorded routes"); return response.json();}),
        loadAtlas(mount.dataset.atlasUrl)]);
      if (data.projectionId !== reference.projectionId) throw Error("The recorded routes do not match the fixed PCA space");
      data.snapshots=data.snapshots.filter(snapshot=>snapshot.update>0);
      const initialBounds=mount.dataset.viewBounds?JSON.parse(mount.dataset.viewBounds):null;
      let viewBounds=initialBounds, selection=null;
      const dragged=new Set();
      const plotRects=new Map();
      const sourcePrediction=data.predictionKind==="source";
      const sourceDigit=data.sourceDigit??0, targetDigit=data.targetDigit??(data.kind==="transport"?5:0);
      const denoisingName=sourcePrediction?"Denoising":"Diffusion";
      const predictionLabel=sourcePrediction?"Predicted source":"Predicted noise";
      let snapshotIndex=Math.max(0,data.snapshots.findIndex(snapshot=>snapshot.update===(data.defaultCheckpoint || 100)));
      let example=0, step=12, animation=0, playing=false, paused=false, playbackId=0;
      const names=data.exampleNames || Array.from({length:data.sampleCount},(_,i)=>`${data.kind==="generator"?"Noise":`Digit ${sourceDigit}`} ${String.fromCharCode(65+i)} → ${targetDigit}`);
      const charts=[...mount.querySelectorAll("canvas[data-method]")];
      const positions=new Map(), hover=new Map(), tooltips=new Map();
      mount.querySelectorAll('[data-digit]').forEach(key=>key.style.setProperty('--digit-color',space.colors[Number(key.dataset.digit)]));
      charts.forEach(canvas=>{
        const tooltip=document.createElement("div");tooltip.className="mnist-route-tooltip mnist-vector-tooltip";tooltip.hidden=true;
        tooltip.setAttribute("role","tooltip");canvas.parentElement.append(tooltip);tooltips.set(canvas,tooltip);
      });
      data.snapshots.forEach((snapshot,i) => control("checkpoint").add(new Option(`${snapshot.update.toLocaleString()} updates`,i)));
      control("checkpoint").value=String(snapshotIndex);
      for(let i=0;i<data.sampleCount;i++) control("example").add(new Option(names[i],i));
      const lastStep=data.solverSteps-1;
      control("step").max=String(lastStep);
      function card(label,id,signal,scale) {
        const figure=document.createElement("figure"), caption=document.createElement("figcaption");
        caption.textContent=label;
        let canvas;
        if(signal) canvas=signalCanvas(signal,scale);
        else {
          canvas=document.createElement("canvas");canvas.width=canvas.height=data.imageSide;
          canvas.setAttribute("role","img");canvas.setAttribute("aria-label",label);
          imageTile(canvas.getContext("2d"),atlas,id,0,0,data.imageSide,data);
        }
        figure.append(caption,canvas);return figure;
      }
      function renderInspector(snapshot,method) {
        const journey=snapshot[method][example], current=journey[step];
        const inspector=document.createElement("article");inspector.className=`mnist-route-inspector ${method}`;
        const heading=document.createElement("h4");heading.textContent=method==="diffusion"?denoisingName:"Flow matching";
        const cards=document.createElement("div");cards.className="mnist-route-cards";
        cards.append(card("Start",journey[0]),card("Now",current));
        cards.append(card(method==="diffusion"?predictionLabel:"Predicted velocity",null,snapshot.predictions[method][example][step]),card("Next",journey[step+1]));
        if(snapshot.targets) cards.append(card(`Paired ${targetDigit}`,snapshot.targets[example]));
        const film=document.createElement("div");film.className="mnist-route-film";
        for(let i=0;i<6;i++) {
          const frame=Math.round(i*lastStep/5), button=document.createElement("button");
          button.type="button";button.setAttribute("aria-label",`${method} step ${frame}`);
          button.setAttribute("aria-pressed",String(Math.round(step/lastStep*5)===i));
          const tile=card(`Step ${frame}`,journey[frame]);button.append(tile);
          button.addEventListener("click",()=>{stop();clearHover();step=frame;render();});film.append(button);
        }
        inspector.append(heading,cards,film);return inspector;
      }
      function renderOverview() {
        const canvas=role("overview");
        if(!canvas||!viewBounds)return;
        const {width,height}=canvas.getBoundingClientRect(), dpr=Math.min(2,window.devicePixelRatio||1);
        canvas.width=Math.max(1,Math.round(width*dpr));canvas.height=Math.max(1,Math.round(height*dpr));
        const ctx=canvas.getContext("2d");ctx.setTransform(dpr,0,0,dpr,0,0);
        const screen=space.frame(ctx,width,height,reference);
        space.reference(ctx,reference,screen,.35);
        const [left,top]=screen([viewBounds[0],viewBounds[3]]), [right,bottom]=screen([viewBounds[1],viewBounds[2]]);
        ctx.fillStyle="rgba(95,109,102,.06)";ctx.fillRect(left,top,right-left,bottom-top);
        ctx.strokeStyle="#5f6d66";ctx.lineWidth=1.5;ctx.setLineDash([4,3]);ctx.strokeRect(left,top,right-left,bottom-top);
        const boundsText=`PC1 ${formatBound(viewBounds[0])}–${formatBound(viewBounds[1])} · PC2 ${formatBound(viewBounds[2])}–${formatBound(viewBounds[3])}`;
        role("zoom-bounds").textContent=boundsText;
        canvas.setAttribute("aria-label",`Full PC1/PC2 map. The outlined rectangle marks ${boundsText}, enlarged in both plots below.`);
        control("reset-zoom").disabled=viewBounds.every((value,i)=>value===initialBounds[i]);
      }
      function renderChart(canvas,snapshot) {
        const {width,height}=canvas.getBoundingClientRect(), dpr=Math.min(2,window.devicePixelRatio||1);
        canvas.width=Math.max(1,Math.round(width*dpr));canvas.height=Math.max(1,Math.round(height*dpr));
        const ctx=canvas.getContext("2d");ctx.setTransform(dpr,0,0,dpr,0,0);
        const screen=space.frame(ctx,width,height,reference,viewBounds);
        plotRects.set(canvas,screen.plotRect);
        ctx.save();
        if(viewBounds) {
          const {left,right,top,bottom}=screen.plotRect;
          ctx.beginPath();ctx.rect(left,top,right-left,bottom-top);ctx.clip();
        }
        const dotOpacity=viewBounds&&data.kind==="transport" ? label => (label===sourceDigit||label===targetDigit ? .2 : .0025) : .2;
        space.reference(ctx,reference,screen,dotOpacity,viewBounds&&data.kind==="transport"?2.75:2);
        const method=canvas.dataset.method, journeys=snapshot[method], color=method==="diffusion"?"#7857b2":"#167d69";
        const allPoints=journeys.map(journey=>journey.map(id=>screen(data.points[id])));
        positions.set(canvas,allPoints);
        const hovered=hover.get(canvas);
        const order=journeys.map((_,i)=>i).filter(i=>i!==example).concat(example);
        const selected=journeys[example];
        const items=[];
        for(let t=0;t<data.solverSteps;t+=3)if(t!==step)
          order.forEach(lane=>items.push({lane,step:t}));
        order.forEach(lane=>items.push({lane,step}));
        items.forEach(item=>{
          const active=item.step===step, chosen=active&&item.lane===example;
          const highlighted=hovered?.lane===item.lane&&hovered?.step===item.step;
          const [x,y]=allPoints[item.lane][item.step], [endX,endY]=allPoints[item.lane][item.step+1];
          const dx=endX-x, dy=endY-y, length=Math.hypot(dx,dy);
          ctx.globalAlpha=chosen||highlighted ? 1 : active ? .7 : .22;
          ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=chosen||highlighted?2.5:active?1.8:1;
          ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(endX,endY);ctx.stroke();
          if(length>.1) {
            const head=Math.min(6,length*.45), ux=dx/length, uy=dy/length;
            ctx.beginPath();ctx.moveTo(endX,endY);
            ctx.lineTo(endX-head*ux+head*.45*uy,endY-head*uy-head*.45*ux);
            ctx.lineTo(endX-head*ux-head*.45*uy,endY-head*uy+head*.45*ux);ctx.closePath();ctx.fill();
          }
          ctx.beginPath();ctx.arc(x,y,chosen?3:active?2:1.2,0,2*Math.PI);ctx.fill();
        });
        ctx.globalAlpha=1;
        ctx.restore();
        if(viewBounds) {
          const {left,right,top,bottom}=screen.plotRect;
          ctx.save();ctx.strokeStyle="#5f6d66";ctx.lineWidth=1;ctx.setLineDash([4,3]);
          ctx.strokeRect(left,top,right-left,bottom-top);ctx.restore();
        }
        if(selection?.canvas===canvas) {
          const {x,y,endX,endY}=selection;
          ctx.save();ctx.fillStyle="rgba(38,121,178,.12)";ctx.strokeStyle="#2679b2";ctx.lineWidth=1.5;ctx.setLineDash([4,3]);
          ctx.fillRect(x,y,endX-x,endY-y);ctx.strokeRect(x,y,endX-x,endY-y);ctx.restore();
        }
        const point=data.points[selected[step]];
        const next=data.points[selected[step+1]];
        const vectorDescription=` Next PC1 ${next[0].toFixed(2)}, PC2 ${next[1].toFixed(2)}.`;
        canvas.setAttribute("aria-label",`${method==="diffusion"?denoisingName:"Flow matching"}, ${names[example]}, step ${step} of ${data.solverSteps}, PC1 ${point[0].toFixed(2)}, PC2 ${point[1].toFixed(2)}.${vectorDescription} Fixed shared PCA axes.${viewBounds?` Zoomed to PC1 ${viewBounds[0]}–${viewBounds[1]}, PC2 ${viewBounds[2]}–${viewBounds[3]}. Drag a rectangle to zoom both plots. Double-click to reset. Press Escape to cancel or reset zoom, plus or minus to zoom about the centre.`:""} Click any route to select it. Use left and right arrows to step through images.`);
      }
      function render() {
        const snapshot=data.snapshots[snapshotIndex];
        charts.forEach(canvas=>renderChart(canvas,snapshot));
        control("step").value=String(step);
        control("example").value=String(example);
        role("step").textContent=`${step} → ${step+1} / ${data.solverSteps}`;

        role("inspectors").replaceChildren(...["diffusion","flow"].map(method=>renderInspector(snapshot,method)));
        updatePlaybackControls();
      }
      function stop() {
        playing=false;paused=false;playbackId++;cancelAnimationFrame(animation);updatePlaybackControls();
      }
      function updatePlaybackControls() {
        control("play").textContent=playing||paused?"Restart route":step===lastStep?"Replay route":"Play route";
        control("pause").textContent=paused?"Resume":"Pause";
        control("pause").disabled=!playing&&!paused;
      }
      function startPlayback(restart) {
        clearHover();
        cancelAnimationFrame(animation);
        const id=++playbackId;
        if(restart || step===lastStep)step=0;
        playing=true;paused=false;updatePlaybackControls();
        const started=performance.now(), initial=step;
        function frame(now) {
          if(!playing || id!==playbackId)return;
          const next=Math.min(lastStep,initial+Math.floor((now-started)/180));
          if(next!==step){step=next;render();}
          if(step===lastStep){stop();return;}
          animation=requestAnimationFrame(frame);
        }
        render();animation=requestAnimationFrame(frame);
      }
      function clearHover() {
        hover.clear();tooltips.forEach(tooltip=>{tooltip.hidden=true;});
        charts.forEach(canvas=>{canvas.style.cursor="crosshair";});
      }
      function formatBound(value) {return Number(value.toPrecision(5)).toString();}
      function cancelSelection() {
        if(!selection)return;
        const {canvas,pointerId}=selection;selection=null;
        if(canvas.hasPointerCapture(pointerId))canvas.releasePointerCapture(pointerId);
      }
      function setZoom(bounds) {
        cancelSelection();clearHover();viewBounds=bounds;
        renderOverview();charts.forEach(canvas=>renderChart(canvas,data.snapshots[snapshotIndex]));
      }
      control("reset-zoom")?.addEventListener("click",()=>setZoom(initialBounds));
      function moveSelection(event) {
        const box=selection.canvas.getBoundingClientRect(), {x,y,rect}=selection;
        const dx=event.clientX-box.left-x, dy=event.clientY-box.top-y;
        selection.endX=Math.max(rect.left,Math.min(rect.right,x+dx));
        selection.endY=Math.max(rect.top,Math.min(rect.bottom,y+dy));
        if(Math.hypot(dx,dy)>=8)dragged.add(selection.canvas);
      }
      control("checkpoint").addEventListener("change",()=>{stop();clearHover();snapshotIndex=Number(control("checkpoint").value);render();});
      control("example").addEventListener("change",()=>{stop();clearHover();example=Number(control("example").value);render();});
      control("step").addEventListener("input",()=>{stop();clearHover();step=Number(control("step").value);render();});

      control("play").addEventListener("click",()=>startPlayback(true));
      control("pause").addEventListener("click",()=>{
        if(paused){startPlayback(false);return;}
        if(!playing)return;
        playing=false;paused=true;playbackId++;cancelAnimationFrame(animation);updatePlaybackControls();
      });
      function routeAt(canvas,event) {
        const box=canvas.getBoundingClientRect(), x=event.clientX-box.left, y=event.clientY-box.top;
        const rect=plotRects.get(canvas);
        if(viewBounds&&rect&&(x<rect.left||x>rect.right||y<rect.top||y>rect.bottom))return null;
        const touch=event.pointerType==="touch" || window.matchMedia("(pointer: coarse)").matches;
        let nearest=null, distance=(touch?20:10)**2;
        (positions.get(canvas)||[]).forEach((points,lane)=>{
          for(let i=0;i<points.length-1;i++) {
            if(i!==step&&i%3!==0)continue;
            const [ax,ay]=points[i], [bx,by]=points[i+1], dx=bx-ax, dy=by-ay;
            const fraction=Math.max(0,Math.min(1,((x-ax)*dx+(y-ay)*dy)/(dx*dx+dy*dy||1)));
            const d=(x-ax-fraction*dx)**2+(y-ay-fraction*dy)**2;
            if(d<distance || (d===distance && lane===example)) {
              nearest={lane,step:i,x,y};distance=d;
            }
          }
        });
        return nearest;
      }
      charts.forEach(canvas=>{
        if(initialBounds) {
          canvas.addEventListener("dblclick",event=>{
            event.preventDefault();setZoom(initialBounds);
          });
          canvas.addEventListener("pointerdown",event=>{
            if(event.button!==0||!event.isPrimary||selection)return;
            dragged.delete(canvas);
            const box=canvas.getBoundingClientRect(), x=event.clientX-box.left, y=event.clientY-box.top, rect=plotRects.get(canvas);
            if(!rect||x<rect.left||x>rect.right||y<rect.top||y>rect.bottom)return;
            stop();clearHover();canvas.focus({preventScroll:true});
            selection={canvas,pointerId:event.pointerId,x,y,endX:x,endY:y,rect};
            canvas.setPointerCapture(event.pointerId);
          });
          canvas.addEventListener("pointerup",event=>{
            if(selection?.canvas!==canvas||selection.pointerId!==event.pointerId)return;
            moveSelection(event);
            const {x,y,endX,endY,rect}=selection;
            if(Math.abs(endX-x)>=8&&Math.abs(endY-y)>=8) {
              const pcX=p=>viewBounds[0]+(p-rect.left)/(rect.right-rect.left)*(viewBounds[1]-viewBounds[0]);
              const pcY=p=>viewBounds[3]-(p-rect.top)/(rect.bottom-rect.top)*(viewBounds[3]-viewBounds[2]);
              const bounds=[pcX(Math.min(x,endX)),pcX(Math.max(x,endX)),pcY(Math.max(y,endY)),pcY(Math.min(y,endY))];
              if(bounds[1]-bounds[0]>=.001&&bounds[3]-bounds[2]>=.001)setZoom(bounds);
            }
            cancelSelection();renderChart(canvas,data.snapshots[snapshotIndex]);
          });
          ["pointercancel","lostpointercapture"].forEach(type=>canvas.addEventListener(type,()=>{
            if(selection?.canvas!==canvas)return;
            dragged.add(canvas);cancelSelection();renderChart(canvas,data.snapshots[snapshotIndex]);
          }));
        }
        canvas.addEventListener("click",event=>{
          if(dragged.delete(canvas))return;
          const nearest=routeAt(canvas,event);
          if(nearest){stop();clearHover();example=nearest.lane;step=nearest.step;render();}
        });
        canvas.addEventListener("pointermove",event=>{
          if(selection) {
            if(selection.canvas===canvas&&selection.pointerId===event.pointerId) {
              moveSelection(event);renderChart(canvas,data.snapshots[snapshotIndex]);
            }
            return;
          }
          if(event.pointerType==="touch")return;
          const nearest=routeAt(canvas,event), previous=hover.get(canvas), tooltip=tooltips.get(canvas);
          if(nearest) {
            hover.set(canvas,nearest);
            {
              const path=data.snapshots[snapshotIndex][canvas.dataset.method][nearest.lane];
              const title=document.createElement("strong");title.textContent=`${names[nearest.lane]} · ${nearest.step} → ${nearest.step+1}`;
              const preview=document.createElement("canvas");preview.width=68;preview.height=32;preview.setAttribute("aria-hidden","true");
              const ctx=preview.getContext("2d");
              imageTile(ctx,atlas,path[nearest.step],0,0,32,data);imageTile(ctx,atlas,path[nearest.step+1],36,0,32,data);
              tooltip.replaceChildren(title,preview);
            }
            tooltip.hidden=false;
            tooltip.style.left=`${Math.max(4,Math.min(canvas.clientWidth-tooltip.offsetWidth-4,nearest.x+12))}px`;
            tooltip.style.top=`${Math.max(canvas.offsetTop+4,canvas.offsetTop+nearest.y-tooltip.offsetHeight-12)}px`;
            canvas.style.cursor="pointer";
          } else {hover.delete(canvas);tooltip.hidden=true;canvas.style.cursor="crosshair";}
          if(previous?.lane!==nearest?.lane||previous?.step!==nearest?.step)renderChart(canvas,data.snapshots[snapshotIndex]);
        });
        canvas.addEventListener("pointerleave",()=>{
          hover.delete(canvas);tooltips.get(canvas).hidden=true;canvas.style.cursor="crosshair";
          renderChart(canvas,data.snapshots[snapshotIndex]);
        });
        canvas.addEventListener("keydown",event=>{
          if(initialBounds&&["Escape","+","=","-"].includes(event.key)) {
            event.preventDefault();
            if(event.key==="Escape") {
              if(selection){dragged.add(selection.canvas);cancelSelection();render();}
              else setZoom(initialBounds);
            } else {
              const factor=event.key==="-"?2:.5;
              const cx=(viewBounds[0]+viewBounds[1])/2, cy=(viewBounds[2]+viewBounds[3])/2;
              const dx=(viewBounds[1]-viewBounds[0])*factor/2, dy=(viewBounds[3]-viewBounds[2])*factor/2;
              if(dx>=.0005&&dx<=1e4)setZoom([cx-dx,cx+dx,cy-dy,cy+dy]);
            }
            return;
          }
          if(!["ArrowLeft","ArrowRight","ArrowUp","ArrowDown","Home","End","Escape"].includes(event.key))return;
          event.preventDefault();stop();clearHover();
          if(event.key==="ArrowUp"||event.key==="ArrowDown") {
            example=(example+(event.key==="ArrowDown"?1:names.length-1))%names.length;render();return;
          }
          if(event.key==="Escape"){render();return;}
          step=event.key==="Home"?0:event.key==="End"?lastStep:Math.max(0,Math.min(lastStep,step+(event.key==="ArrowRight"?1:-1)));
          render();
        });
      });
      mount.querySelector(".mnist-route-content").hidden=false;
      status.textContent="";status.hidden=true;
      const resizeObserver=new ResizeObserver(()=>{cancelSelection();clearHover();renderOverview();charts.forEach(canvas=>renderChart(canvas,data.snapshots[snapshotIndex]));});
      charts.forEach(canvas=>resizeObserver.observe(canvas));
      if(role("overview"))resizeObserver.observe(role("overview"));
      renderOverview();render();updatePlaybackControls();
    } catch(error) {
      status.hidden=false;
      status.textContent="The recorded routes couldn’t load. Refresh the page to try again.";
      console.error(error);
    }
  }));
}());
