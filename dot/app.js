var canvas = document.getElementById('canvas');
var ctx = canvas.getContext('2d');
var raf;
var doFlash = false;
var flash = true;

var ball = {
    x: 100,
    y: 100,
    vx: 5,
    vy: 2,
    radius: 7,
    color: 'black',
    draw: function() {
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI*2, true);
        ctx.closePath();
        ctx.fillStyle = this.color;
        ctx.fill();
    }
};

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ball.draw();
    ball.x += ball.vx;
    ball.y += ball.vy;
    if (ball.y + ball.vy > canvas.height || ball.y + ball.vy < 0) {
        ball.vy = -ball.vy;
        window.requestAnimationFrame(metaFlash);
        return;
    }
    if (ball.x + ball.vx > canvas.width || ball.x + ball.vx < 0) {
        ball.vx = -ball.vx;
        window.requestAnimationFrame(metaFlash);
        return;
    }
    // It setting up...
    flash = !flash;
    width = 5;
    if (doFlash && flash) {
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(canvas.width, 0); // TODO hardcoded size
        ctx.lineWidth = width;
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(canvas.wdith, 0);
        ctx.lineTo(canvas.width, canvas.height);
        ctx.lineWidth = width;
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(canvas.width, canvas.height);
        ctx.lineTo(0, canvas.height);
        ctx.lineWidth = width;
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, canvas.height);
        ctx.lineTo(0, 0);
        ctx.lineWidth = width;
        ctx.stroke();
    }

    $.post({url: 'writer.php',
            data: {t: window.performance.now(),
                    x: ball.x / 3,
                    y: ball.y / 3
            }
    });
    raf = window.requestAnimationFrame(draw);
}

function metaFlash() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.moveTo(canvas.width/2, 0);
    ctx.lineTo(canvas.width/2, canvas.height);
    ctx.lineWidth = 5;
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.lineWidth = 5;
    ctx.stroke();
    $.post({url: 'writer.php',
            data: {t: window.performance.now(),
                    x: -1,
                    y: -1
            }
    });
    window.requestAnimationFrame(draw);
}

function onclick_flash() {
    if (doFlash) {
        doFlash = false;
    } else {
        doFlash = true;
    }
}

draw();
