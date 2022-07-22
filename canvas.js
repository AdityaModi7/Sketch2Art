// create canvas element and append it to document body
var canvas = document.getElementById('sketchBoard');

// get canvas 2D context and set him correct size
var ctx = canvas.getContext('2d');
ctx.canvas.width = canvas.width;
ctx.canvas.height = canvas.height;



// last known position
var pos = { x: 0, y: 0 };
var color = 'black';

ctx.lineWidth = 2;
ctx.lineCap = 'round';

canvas.addEventListener('mousedown', setPosition);
canvas.addEventListener('mouseenter', setPosition);

canvas.addEventListener('mousemove', draw);

// new position from mouse event
function setPosition(e) {
    var rect = canvas.getBoundingClientRect();
    pos.x = (e.clientX - canvas.offsetLeft) * (canvas.width / rect.width);
    pos.y = (e.clientY - canvas.offsetTop) * (canvas.height / rect.height);
}

function draw(e) {
    if (e.buttons !== 1) return;
    ctx.strokeStyle = color;

    ctx.beginPath(); // begin

    ctx.moveTo(pos.x, pos.y); // from
    setPosition(e);
    ctx.lineTo(pos.x, pos.y); // to

    ctx.stroke(); // draw
}