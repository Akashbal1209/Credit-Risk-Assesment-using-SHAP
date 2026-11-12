/* Live mode: calls /predict on the same server. Set DEMO = true to preview with the stand-in formula. */
const DEMO=false, API_URL="";
const $=s=>document.querySelector(s);
const st={grade:"B",home:"RENT",def:"N"};
const opts={home:[["Rent","RENT"],["Own","OWN"],["Mortgage","MORTGAGE"]],def:[["No","N"],["Yes","Y"]]};
document.querySelectorAll(".chips").forEach(box=>{const k=box.dataset.k;opts[k].forEach(([l,v])=>{v=v||l;const b=document.createElement("button");b.type="button";b.textContent=l;if(st[k]===v)b.classList.add("on");b.onclick=()=>{st[k]=v;[...box.children].forEach(c=>c.classList.toggle("on",c===b))};box.appendChild(b)})});

/* background particles */
const cv=$("#cv"),cx=cv.getContext("2d"),still=matchMedia("(prefers-reduced-motion:reduce)").matches;
let W,H,P=[],mx=0,my=0,col=[99,102,241],tg=[99,102,241];
function fit(){const d=Math.min(devicePixelRatio||1,2);W=innerWidth;H=innerHeight;cv.width=W*d;cv.height=H*d;cx.setTransform(d,0,0,d,0,0);P=Array.from({length:Math.round(W*H/16000)+22},()=>({x:Math.random()*W,y:Math.random()*H,vx:(Math.random()-.5)*(still?0:.35),vy:(Math.random()-.5)*(still?0:.35),z:Math.random()*.8+.2}))}
fit();addEventListener("resize",fit);
(function loop(){cx.clearRect(0,0,W,H);for(let i=0;i<3;i++)col[i]+=(tg[i]-col[i])*.04;const c=col.map(Math.round).join(",");
P.forEach(p=>{p.x+=p.vx;p.y+=p.vy;if(p.x<0||p.x>W)p.vx*=-1;if(p.y<0||p.y>H)p.vy*=-1;p.px=p.x+mx*42*p.z;p.py=p.y+my*42*p.z});
for(let i=0;i<P.length;i++){const a=P[i];cx.fillStyle="rgba("+c+","+(.5+a.z*.5)+")";cx.beginPath();cx.arc(a.px,a.py,a.z*1.8+.4,0,7);cx.fill();
for(let j=i+1;j<P.length;j++){const b=P[j],d=Math.hypot(a.px-b.px,a.py-b.py);if(d<130){cx.strokeStyle="rgba("+c+","+(.28*(1-d/130))+")";cx.beginPath();cx.moveTo(a.px,a.py);cx.lineTo(b.px,b.py);cx.stroke()}}}
requestAnimationFrame(loop)})();

/* tilt + glare (fine pointers only) */
const card=$("#card"),fine=matchMedia("(pointer:fine)").matches;
if(!fine)$("#hint").textContent="";
addEventListener("pointermove",e=>{mx=e.clientX/W-.5;my=e.clientY/H-.5;if(!fine||still)return;const r=card.getBoundingClientRect(),x=(e.clientX-r.left)/r.width,y=(e.clientY-r.top)/r.height;
card.style.transform="rotateY("+((x-.5)*10)+"deg) rotateX("+((.5-y)*10)+"deg)";card.style.setProperty("--gx",x*100+"%");card.style.setProperty("--gy",y*100+"%")});
document.addEventListener("pointerleave",()=>{card.style.transform=""});

/* data */
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
function payload(){const n=id=>parseFloat($("#"+id).value),inc=n("inc"),amt=n("amt");
return{person_age:Math.round(n("age")),person_income:inc,person_home_ownership:st.home,person_emp_length:n("emp"),loan_intent:$("#intent").value,loan_grade:st.grade,loan_amnt:amt,loan_int_rate:n("rate"),loan_percent_income:+(amt/inc).toFixed(2),cb_person_default_on_file:st.def,cb_person_cred_hist_length:Math.round(n("cred"))}}
function mock(p){const g="ABCDEFG".indexOf(p.loan_grade)/6,pti=Math.min(1,p.loan_percent_income/.5),r=Math.min(1,(p.loan_int_rate-5)/19);
let x=.06+.38*g+.28*pti+.2*r+(p.person_home_ownership==="RENT"?.05:0)+(p.cb_person_default_on_file==="Y"?.08:0)-(p.person_emp_length>5?.04:0);
x=Math.max(.02,Math.min(.97,x));const t=.38;return{default_probability:x,default_prediction:+(x>=t),threshold:t,Result:x>=t?"High Risk":"Low Risk"}}
async function call(p,note){if(DEMO){await sleep(600);return mock(p)}
for(let i=0;i<4;i++){const ctl=new AbortController(),to=setTimeout(()=>ctl.abort(),45000),wake=setTimeout(()=>note("Server is waking up. This can take up to a minute…"),4000);
try{const r=await fetch(API_URL+"/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(p),signal:ctl.signal});clearTimeout(to);clearTimeout(wake);
if(r.status===422)throw new Error("bad");if(!r.ok)throw new Error("srv");return await r.json()}
catch(e){clearTimeout(to);clearTimeout(wake);if(e.message==="bad")throw e;if(i<3){note("Retrying ("+(i+2)+" of 4)…");await sleep(2500)}}}
throw new Error("down")}

/* submit */
const form=$("#form"),msg=$("#msg"),go=$("#go"),steps=["Reading profile","Scoring 11 features","Applying risk threshold"];
form.addEventListener("submit",async e=>{e.preventDefault();msg.className="msg";msg.textContent="";
let ok=true;["age","inc","amt","rate","emp","cred"].forEach(id=>{const el=$("#"+id),v=parseFloat(el.value),bad=!(v>=0)||(["age","inc","amt"].includes(id)&&v<=0);el.classList.toggle("bad",bad);if(bad)ok=false});
if(!ok){msg.className="msg err";msg.textContent="Check the highlighted fields. Age, income, and loan need a value above 0.";return}
const p=payload();go.disabled=true;card.classList.add("run");let k=0;go.textContent=steps[0]+"…";
const tick=setInterval(()=>{k=Math.min(k+1,2);go.textContent=steps[k]+"…"},900);
try{const [d]=await Promise.all([call(p,t=>msg.textContent=t),sleep(2700)]);show(d,p)}
catch(err){msg.className="msg err";msg.textContent=err.message==="bad"?"The server rejected some values. Check your inputs and try again.":"Couldn’t reach the server. Try again in a moment."}
finally{clearInterval(tick);card.classList.remove("run");go.disabled=false;go.textContent="Analyze application"}});

function show(d,p){const hi=d.default_prediction===1,pr=d.default_probability;
document.body.className=hi?"high":"low";tg=hi?[240,96,93]:[47,191,143];
form.classList.add("hide");const res=$("#res");res.classList.remove("hide");res.classList.add("view");
$("#vd").textContent=d.Result;$("#vd").style.color=hi?"var(--bad)":"var(--ok)";
const rv=$("#rv");rv.style.stroke=hi?"var(--bad)":"var(--ok)";
const f=[["Loan to income",Math.min(1,p.loan_percent_income/.5)],["Interest rate",Math.min(1,(p.loan_int_rate-5)/19)]];
$("#fx").innerHTML=f.map(([n,v])=>'<div class="f"><span>'+n+'</span><i><s data-w="'+Math.round(v*100)+'"></s></i><em>'+Math.round(v*100)+'</em></div>').join("");
requestAnimationFrame(()=>{rv.style.strokeDashoffset=339.3*(1-pr);document.querySelectorAll(".f s").forEach(s=>s.style.width=s.dataset.w+"%")});
const t0=performance.now();(function c(t){const k=still?1:Math.min(1,(t-t0)/1300);$("#pc").textContent=(pr*100*k).toFixed(1)+"%";if(k<1)requestAnimationFrame(c)})(t0)}
$("#back").onclick=()=>{$("#res").classList.add("hide");$("#res").classList.remove("view");form.classList.remove("hide");form.classList.remove("view");void form.offsetWidth;form.classList.add("view");document.body.className="";tg=[99,102,241];$("#rv").style.strokeDashoffset=339.3};
