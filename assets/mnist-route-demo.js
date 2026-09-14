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
      let snapshotIndex=Math.max(0,data.snapshots.findIndex(snapshot=>snapshot.update===(data.defaultCheckpoint || 100)));
      let example=0, step=0, animation=0, playing=false, paused=false, playbackId=0;
      const names=data.exampleNames || Array.from({length:data.sampleCount},(_,i)=>`${data.kind==="generator"?"Noise":"Zero"} ${String.fromCharCode(65+i)} → ${data.kind==="generator"?"0":"5"}`);
      const charts=[...mount.querySelectorAll("canvas[data-method]")];
      const positions=new Map(), hover=new Map(), tooltips=new Map();
      charts.forEach(canvas=>{
        const tooltip=document.createElement("div");tooltip.className="mnist-route-tooltip";tooltip.hidden=true;
        tooltip.setAttribute("role","tooltip");canvas.parentElement.append(tooltip);tooltips.set(canvas,tooltip);
      });
      data.snapshots.forEach((snapshot,i) => control("checkpoint").add(new Option(snapshot.update ? `${snapshot.update.toLocaleString()} updates` : "Untrained",i)));
      control("checkpoint").value=String(snapshotIndex);
      for(let i=0;i<data.sampleCount;i++) control("example").add(new Option(names[i],i));
      control("step").max=String(data.solverSteps);
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
      let renderedTargetSnapshot=null, renderedTargetTime=null;
      function renderTargets(snapshot) {
        if (!snapshot.training) return;
        const time=Number(control("training-time").value);
        if(renderedTargetSnapshot===snapshot && renderedTargetTime===time)return;
        renderedTargetSnapshot=snapshot;renderedTargetTime=time;
        const record=snapshot.training[time];
        const target=role("targets");target.replaceChildren();
        for(const method of ["diffusion","flow"]) {
          const box=document.createElement("article"), title=document.createElement("h4");
          title.textContent=method==="diffusion" ? `Diffusion · τ = ${record.tau.toFixed(2)}` : `Flow matching · t = ${record.t.toFixed(2)}`;
          const cards=document.createElement("div");cards.className="mnist-route-cards";
          const signal=record[method==="diffusion" ? "noise" : "velocity"];
          const scale=Array.isArray(signal)?Math.max(.001,...signal.map(Math.abs)):signal.scale;
          cards.append(card("Clean digit",record.clean),card("Input",record[`${method}Input`]),
            card(method==="diffusion"?"Target noise":"Target velocity",null,signal,scale),
            card("Prediction",null,record[`${method}Prediction`],scale));
          box.append(title,cards);target.append(box);
        }
      }
      function renderInspector(snapshot,method) {
        const journey=snapshot[method][example], current=journey[step], point=data.points[current];
        const inspector=document.createElement("article");inspector.className=`mnist-route-inspector ${method}`;
        const heading=document.createElement("h4");heading.textContent=method==="diffusion"?"Diffusion: one step":"Flow matching: one step";
        const coordinate=document.createElement("p");coordinate.className="mnist-route-coordinates";
        coordinate.textContent=`${names[example]} · PC1 ${point[0].toFixed(2)} · PC2 ${point[1].toFixed(2)}`;
        const cards=document.createElement("div");cards.className="mnist-route-cards";
        cards.append(card("Start",journey[0]),card("Now",current));
        if(step<data.solverSteps) {
          cards.append(card(method==="diffusion"?"Predicted noise":"Predicted velocity",null,snapshot.predictions[method][example][step]),card("Next",journey[step+1]));
        } else cards.append(card("Finished",current));
        if(snapshot.targets) cards.append(card(method==="diffusion"?"Training example":"Paired target",snapshot.targets[example]));
        const film=document.createElement("div");film.className="mnist-route-film";
        for(let i=0;i<6;i++) {
          const frame=Math.round(i*data.solverSteps/5), button=document.createElement("button");
          button.type="button";button.setAttribute("aria-label",`${method} step ${frame}`);
          button.setAttribute("aria-pressed",String(Math.round(step/data.solverSteps*5)===i));
          const tile=card(`Step ${frame}`,journey[frame]);button.append(tile);
          button.addEventListener("click",()=>{stop();step=frame;render();});film.append(button);
        }
        inspector.append(heading,coordinate,cards,film);return inspector;
      }
      function renderChart(canvas,snapshot) {
        const {width,height}=canvas.getBoundingClientRect(), dpr=Math.min(2,window.devicePixelRatio||1);
        canvas.width=Math.max(1,Math.round(width*dpr));canvas.height=Math.max(1,Math.round(height*dpr));
        const ctx=canvas.getContext("2d");ctx.setTransform(dpr,0,0,dpr,0,0);
        const screen=space.frame(ctx,width,height,reference);
        space.reference(ctx,reference,screen,.2);
        const method=canvas.dataset.method, journeys=snapshot[method], color=method==="diffusion"?"#7857b2":"#167d69";
        const allPoints=journeys.map(journey=>journey.map(id=>screen(data.points[id])));
        positions.set(canvas,allPoints);
        const hovered=hover.get(canvas);
        const order=journeys.map((_,i)=>i).filter(i=>i!==example).concat(example);
        order.forEach(lane=>{
          ctx.globalAlpha=lane === example ? 1 : lane === hovered?.lane ? .65 : .3;
          ctx.lineWidth=lane === example || lane === hovered?.lane ? 2.5 : 1.3;ctx.strokeStyle=color;
          ctx.beginPath();allPoints[lane].forEach(([x,y],i)=>{if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();
        });
        ctx.globalAlpha=1;
        const selected=journeys[example], points=allPoints[example];
        ctx.fillStyle=color;
        points.forEach(([x,y])=>{ctx.beginPath();ctx.arc(x,y,2,0,2*Math.PI);ctx.fill();});
        const [x,y]=points[step];
        ctx.strokeStyle=color;ctx.lineWidth=2;ctx.strokeRect(x-13,y-13,26,26);
        imageTile(ctx,atlas,selected[step],x-11,y-11,22,data);
        const point=data.points[selected[step]];
        canvas.setAttribute("aria-label",`${method==="diffusion"?"Diffusion":"Flow matching"}, ${names[example]}, step ${step} of ${data.solverSteps}, PC1 ${point[0].toFixed(2)}, PC2 ${point[1].toFixed(2)}. Fixed shared PCA axes. Click any route to select it. Use left and right arrows to step through images.`);
      }
      function render() {
        const snapshot=data.snapshots[snapshotIndex];
        charts.forEach(canvas=>renderChart(canvas,snapshot));
        control("step").value=String(step);
        control("example").value=String(example);
        role("step").textContent=`${step} / ${data.solverSteps} steps`;
        role("loss").textContent=snapshot.update===0 ? "Untrained teaching models" : `Recorded training MSE · diffusion ${snapshot.diffusionLoss.toFixed(4)} · flow ${snapshot.flowLoss.toFixed(4)}`;
        role("inspectors").replaceChildren(...["diffusion","flow"].map(method=>renderInspector(snapshot,method)));
        renderTargets(snapshot);updatePlaybackControls();
      }
      function stop() {
        playing=false;paused=false;playbackId++;cancelAnimationFrame(animation);updatePlaybackControls();
      }
      function updatePlaybackControls() {
        control("play").textContent=playing||paused?"Restart route":step===data.solverSteps?"Replay route":"Play route";
        control("pause").textContent=paused?"Resume":"Pause";
        control("pause").disabled=!playing&&!paused;
      }
      function startPlayback(restart) {
        cancelAnimationFrame(animation);
        const id=++playbackId;
        if(restart || step===data.solverSteps)step=0;
        playing=true;paused=false;updatePlaybackControls();
        const started=performance.now(), initial=step;
        function frame(now) {
          if(!playing || id!==playbackId)return;
          const next=Math.min(data.solverSteps,initial+Math.floor((now-started)/180));
          if(next!==step){step=next;render();}
          if(step===data.solverSteps){stop();return;}
          animation=requestAnimationFrame(frame);
        }
        render();animation=requestAnimationFrame(frame);
      }
      function clearHover() {
        hover.clear();tooltips.forEach(tooltip=>{tooltip.hidden=true;});
        charts.forEach(canvas=>{canvas.style.cursor="crosshair";});
      }
      control("checkpoint").addEventListener("change",()=>{stop();clearHover();snapshotIndex=Number(control("checkpoint").value);render();});
      control("example").addEventListener("change",()=>{stop();example=Number(control("example").value);render();});
      control("step").addEventListener("input",()=>{stop();step=Number(control("step").value);render();});
      if(control("training-time")) control("training-time").addEventListener("change",()=>renderTargets(data.snapshots[snapshotIndex]));
      control("play").addEventListener("click",()=>startPlayback(true));
      control("pause").addEventListener("click",()=>{
        if(paused){startPlayback(false);return;}
        if(!playing)return;
        playing=false;paused=true;playbackId++;cancelAnimationFrame(animation);updatePlaybackControls();
      });
      function routeAt(canvas,event) {
        const box=canvas.getBoundingClientRect(), x=event.clientX-box.left, y=event.clientY-box.top;
        const touch=event.pointerType==="touch" || window.matchMedia("(pointer: coarse)").matches;
        let nearest=null, distance=(touch?20:10)**2;
        (positions.get(canvas)||[]).forEach((points,lane)=>{
          for(let i=0;i<points.length-1;i++) {
            const [ax,ay]=points[i], [bx,by]=points[i+1], dx=bx-ax, dy=by-ay;
            const fraction=Math.max(0,Math.min(1,((x-ax)*dx+(y-ay)*dy)/(dx*dx+dy*dy||1)));
            const d=(x-ax-fraction*dx)**2+(y-ay-fraction*dy)**2;
            if(d<distance || (d===distance && lane===example)) {
              nearest={lane,step:i+(fraction>=.5?1:0),x,y};distance=d;
            }
          }
        });
        return nearest;
      }
      charts.forEach(canvas=>{
        canvas.addEventListener("click",event=>{
          const nearest=routeAt(canvas,event);
          if(nearest){stop();example=nearest.lane;step=nearest.step;render();}
        });
        canvas.addEventListener("pointermove",event=>{
          if(event.pointerType==="touch")return;
          const nearest=routeAt(canvas,event), previous=hover.get(canvas), tooltip=tooltips.get(canvas);
          if(nearest) {
            hover.set(canvas,nearest);tooltip.textContent=`${names[nearest.lane]} · step ${nearest.step}`;tooltip.hidden=false;
            tooltip.style.left=`${Math.max(4,Math.min(canvas.clientWidth-tooltip.offsetWidth-4,nearest.x+12))}px`;
            tooltip.style.top=`${Math.max(canvas.offsetTop+4,canvas.offsetTop+nearest.y-tooltip.offsetHeight-12)}px`;
            canvas.style.cursor="pointer";
          } else {hover.delete(canvas);tooltip.hidden=true;canvas.style.cursor="crosshair";}
          if(previous?.lane!==nearest?.lane)renderChart(canvas,data.snapshots[snapshotIndex]);
        });
        canvas.addEventListener("pointerleave",()=>{
          hover.delete(canvas);tooltips.get(canvas).hidden=true;canvas.style.cursor="crosshair";
          renderChart(canvas,data.snapshots[snapshotIndex]);
        });
        canvas.addEventListener("keydown",event=>{
          if(!["ArrowLeft","ArrowRight","Home","End"].includes(event.key))return;
          event.preventDefault();stop();
          step=event.key==="Home"?0:event.key==="End"?data.solverSteps:Math.max(0,Math.min(data.solverSteps,step+(event.key==="ArrowRight"?1:-1)));
          render();
        });
      });
      mount.querySelector(".mnist-route-content").hidden=false;
      status.textContent="Recorded offline · fixed PC1/PC2";
      const resizeObserver=new ResizeObserver(()=>{clearHover();charts.forEach(canvas=>renderChart(canvas,data.snapshots[snapshotIndex]));});
      charts.forEach(canvas=>resizeObserver.observe(canvas));
      render();updatePlaybackControls();
    } catch(error) {
      status.textContent="The recorded routes couldn’t load. Refresh the page to try again.";
      console.error(error);
    }
  }));
}());
