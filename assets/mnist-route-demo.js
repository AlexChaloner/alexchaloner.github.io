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
    const canvas = document.createElement("canvas"); canvas.width = canvas.height = 8;
    canvas.setAttribute("role", "img"); canvas.setAttribute("aria-label", "Signed pixel signal: blue positive, orange negative");
    const ctx = canvas.getContext("2d"), pixels = ctx.createImageData(8,8);
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
      let snapshotIndex=data.snapshots.length-1, example=0, step=0, animation=0, playing=false;
      const charts=[...mount.querySelectorAll("canvas[data-method]")];
      const positions=new Map();
      data.snapshots.forEach((snapshot,i) => control("checkpoint").add(new Option(snapshot.update ? `${snapshot.update.toLocaleString()} updates` : "Untrained",i)));
      control("checkpoint").value=String(snapshotIndex);
      for(let i=0;i<data.sampleCount;i++) control("example").add(new Option(`Example ${i+1}`,i));
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
      function renderTargets(snapshot) {
        if (!snapshot.training) return;
        const record=snapshot.training[Number(control("training-time").value)];
        const target=role("targets");target.replaceChildren();
        for(const method of ["diffusion","flow"]) {
          const box=document.createElement("article"), title=document.createElement("h4");
          title.textContent=method==="diffusion" ? `Diffusion · τ = ${record.tau.toFixed(2)}` : `Flow matching · t = ${record.t.toFixed(2)}`;
          const cards=document.createElement("div");cards.className="mnist-route-cards";
          const signal=record[method==="diffusion" ? "noise" : "velocity"];
          const scale=Math.max(.001,...signal.map(Math.abs));
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
        coordinate.textContent=`PC1 ${point[0].toFixed(2)} · PC2 ${point[1].toFixed(2)}`;
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
        journeys.forEach((journey,lane)=>{
          ctx.globalAlpha=lane===example?1:.2;ctx.lineWidth=lane===example?2.5:1;ctx.strokeStyle=color;
          ctx.beginPath();journey.forEach((id,i)=>{const [x,y]=screen(data.points[id]);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();
        });
        ctx.globalAlpha=1;
        const selected=journeys[example], points=selected.map(id=>screen(data.points[id]));positions.set(canvas,points);
        ctx.fillStyle=color;
        points.forEach(([x,y])=>{ctx.beginPath();ctx.arc(x,y,2,0,2*Math.PI);ctx.fill();});
        const [x,y]=points[step];
        ctx.strokeStyle=color;ctx.lineWidth=2;ctx.strokeRect(x-13,y-13,26,26);
        imageTile(ctx,atlas,selected[step],x-11,y-11,22,data);
        const point=data.points[selected[step]];
        canvas.setAttribute("aria-label",`${method==="diffusion"?"Diffusion":"Flow matching"}, example ${example+1}, step ${step} of ${data.solverSteps}, PC1 ${point[0].toFixed(2)}, PC2 ${point[1].toFixed(2)}. Fixed shared PCA axes. Use left and right arrows to step through images.`);
      }
      function render() {
        const snapshot=data.snapshots[snapshotIndex];
        charts.forEach(canvas=>renderChart(canvas,snapshot));
        control("step").value=String(step);
        role("step").textContent=`${step} / ${data.solverSteps} steps`;
        role("loss").textContent=snapshot.update===0 ? "Untrained teaching models" : `Recorded training MSE · diffusion ${snapshot.diffusionLoss.toFixed(4)} · flow ${snapshot.flowLoss.toFixed(4)}`;
        role("inspectors").replaceChildren(...["diffusion","flow"].map(method=>renderInspector(snapshot,method)));
        renderTargets(snapshot);
      }
      function stop() {
        playing=false;cancelAnimationFrame(animation);control("play").textContent="Play route";
      }
      control("checkpoint").addEventListener("change",()=>{stop();snapshotIndex=Number(control("checkpoint").value);render();});
      control("example").addEventListener("change",()=>{stop();example=Number(control("example").value);render();});
      control("step").addEventListener("input",()=>{stop();step=Number(control("step").value);render();});
      if(control("training-time")) control("training-time").addEventListener("change",()=>renderTargets(data.snapshots[snapshotIndex]));
      control("play").addEventListener("click",()=>{
        if(playing){stop();return;}
        if(step===data.solverSteps)step=0;
        playing=true;control("play").textContent="Pause route";
        const started=performance.now(), initial=step;
        function frame(now) {
          if(!playing)return;
          const next=Math.min(data.solverSteps,initial+Math.floor((now-started)/180));
          if(next!==step){step=next;render();}
          if(step===data.solverSteps){stop();return;}
          animation=requestAnimationFrame(frame);
        }
        render();animation=requestAnimationFrame(frame);
      });
      charts.forEach(canvas=>{
        canvas.addEventListener("click",event=>{
          const box=canvas.getBoundingClientRect(), x=event.clientX-box.left, y=event.clientY-box.top;
          let nearest=-1, distance=24**2;
          positions.get(canvas).forEach(([px,py],i)=>{const d=(x-px)**2+(y-py)**2;if(d<distance){nearest=i;distance=d;}});
          if(nearest>=0){stop();step=nearest;render();}
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
      new ResizeObserver(()=>charts.forEach(canvas=>renderChart(canvas,data.snapshots[snapshotIndex]))).observe(mount);
      render();
    } catch(error) {
      status.textContent="The recorded routes couldn’t load. Refresh the page to try again.";
      console.error(error);
    }
  }));
}());
