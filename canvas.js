var canvas = document.getElementById('sketchBoard');
var container = document.getElementById('container');
canvas.width = 256;
canvas.height = 256;

// get canvas 2D context and set its correct size
var ctx = canvas.getContext('2d');
ctx.canvas.width = canvas.width;
ctx.canvas.height = canvas.height;

// last known position
var pos = { x: 0, y: 0 };
var color = 'black';
var isSketching = false;

ctx.lineWidth = 1;
ctx.lineCap = 'round';

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseenter', setPosition);
canvas.addEventListener('mousemove', draw);

// new position from mouse event
function setPosition(e) {
    var rect = canvas.getBoundingClientRect();
    pos.x = (e.clientX - canvas.offsetLeft) * (canvas.width / rect.width);
    pos.y = (e.clientY - (canvas.offsetTop + container.offsetTop) + container.scrollTop) * (canvas.height / rect.height);
}

function startDrawing() {
    if (isSketching) {
        isDrawing = true;
        setPosition(event);
    }
}

function stopDrawing() {
    isDrawing = false;
}

function draw(e) {
    ctx.fillStyle = "white";
    if (isErasing) {
        if (e.buttons !== 1) return;
        else if (e.buttons !== 1 || !isDrawing) return;
        ctx.strokeStyle = ctx.fillStyle;
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
        setPosition(e);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
    }
     else {
        if (e.buttons !== 1 || !isDrawing) return;
        ctx.strokeStyle = color;
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
        setPosition(e);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
    }
}

function togglePencil() {
    isDrawing = !isDrawing;
    pencilButton.classList.toggle('active');
    if (isDrawing) {
        eraserButton.classList.remove('active');
        isErasing = false;
    }
}

function toggleEraser() {
    isErasing = !isErasing;
    eraserButton.classList.toggle('active');
    if (isErasing) {
        pencilButton.classList.remove('active');
        isDrawing = false;
    }
}

function refresh() {
    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear the canvas
}

var resetButton = document.getElementById('resetButton');
resetButton.addEventListener('click', refresh);

var pencilButton = document.getElementById('pencilButton');
pencilButton.addEventListener('click', togglePencil); {
    isSketching = !isSketching;
}

var eraserButton = document.getElementById('eraserButton');
eraserButton.addEventListener('click', toggleEraser);
