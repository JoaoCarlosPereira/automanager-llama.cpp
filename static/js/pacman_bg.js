/*!
* Start Bootstrap - Personal v1.0.1 (https://startbootstrap.com/template-overviews/personal)
* Copyright 2013-2023 Start Bootstrap
* Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-personal/blob/master/LICENSE)
*/
document.addEventListener('DOMContentLoaded', function () {
    const canvas = document.getElementById('pacman-background');

    if (!canvas) {
        return;
    }

    const ctx = canvas.getContext('2d');
    const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    let width = 0;
    let height = 0;
    let dpr = Math.min(window.devicePixelRatio || 1, 2);
    let lanes = [];
    let pellets = [];
    let ghosts = [];
    let sparks = [];
    let codeSymbols = [];
    let circuitNodes = [];
    let terminalBlocks = [];
    let pacman = null;
    let animationId = 0;
    let lastFrame = 0;
    
    // Estados do Fundo
    let powerModeTimer = 0;
    let bgScore = 0;

    function resizeCanvas() {
        width = window.innerWidth;
        height = window.innerHeight;
        dpr = Math.min(window.devicePixelRatio || 1, 2);
        canvas.width = Math.floor(width * dpr);
        canvas.height = Math.floor(height * dpr);
        canvas.style.width = width + 'px';
        canvas.style.height = height + 'px';
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        buildScene();
    }

    function buildScene() {
        const laneCount = Math.max(5, Math.min(8, Math.round(height / 150)));
        const spacing = height / (laneCount + 1);
        lanes = [];
        pellets = [];
        ghosts = [];
        sparks = [];
        codeSymbols = [];
        circuitNodes = [];
        terminalBlocks = [];

        let bigPelletCount = 0;

        for (let i = 0; i < laneCount; i++) {
            const y = spacing * (i + 1);
            const amplitude = 10 + (i % 3) * 5;
            const speed = 34 + i * 4;
            const phase = i * 0.8;
            lanes.push({ y, amplitude, speed, phase });

            const pelletGap = 180 + (i % 2) * 40;
            const pelletCount = Math.ceil((width + 400) / pelletGap);

            for (let j = 0; j < pelletCount; j++) {
                let isBig = false;
                if (bigPelletCount < 3 && Math.random() > 0.95) {
                    isBig = true;
                    bigPelletCount++;
                }

                pellets.push({
                    lane: i,
                    x: j * pelletGap - 200,
                    baseY: y,
                    radius: isBig ? 6 : 3.1,
                    alpha: isBig ? 1.0 : 0.75,
                    offset: (j % 4) * 0.9,
                    eaten: false,
                    respawnTimer: 0
                });
            }
        }

        const activeLane = Math.floor(laneCount * 0.5);
        pacman = {
            lane: activeLane,
            x: -60,
            y: lanes[activeLane].y,
            radius: 18,
            speed: Math.max(85, width * 0.085),
            direction: 1,
            mouth: 0,
            mouthSpeed: 7.5,
            dying: false,
            deathTimer: 0
        };

        const ghostPalette = [
            { body: '#ff4d6d', eye: '#d9f6ff' },
            { body: '#34d3ff', eye: '#ffffff' },
            { body: '#ffb703', eye: '#ffffff' },
            { body: '#b582ff', eye: '#ffffff' },
            { body: '#00f5d4', eye: '#ffffff' }
        ];

        for (let i = 0; i < 5; i++) {
            const laneIndex = i % lanes.length;
            ghosts.push({
                lane: laneIndex,
                x: width + 200 + i * 350,
                y: lanes[laneIndex].y,
                width: 28,
                height: 28,
                speed: 55 + i * 5,
                direction: -1,
                bob: i * 1.5,
                color: ghostPalette[i % ghostPalette.length],
                eaten: false
            });
        }

        const glyphs = ['</>', '{ }', '[]', '()', '01', '=>', 'fn', 'SQL'];
        const symbolCount = Math.max(14, Math.round((width * height) / 52000));
        const nodeCount = Math.max(8, Math.round(width / 180));
        const terminalCount = Math.max(3, Math.round(width / 420));

        for (let i = 0; i < symbolCount; i++) {
            codeSymbols.push({
                text: glyphs[i % glyphs.length],
                x: Math.random() * width,
                y: Math.random() * height,
                size: 11 + Math.random() * 9,
                alpha: 0.08 + Math.random() * 0.1,
                drift: 0.2 + Math.random() * 0.55,
                phase: Math.random() * Math.PI * 2
            });
        }

        for (let i = 0; i < nodeCount; i++) {
            const x = (width / (nodeCount + 1)) * (i + 1);
            circuitNodes.push({
                x, y: height * (0.18 + (i % 4) * 0.17),
                pulse: Math.random() * Math.PI * 2,
                width: 80 + Math.random() * 90
            });
        }

        for (let i = 0; i < terminalCount; i++) {
            terminalBlocks.push({
                x: 40 + i * ((width - 80) / terminalCount),
                y: 40 + (i % 2) * 58,
                width: 88 + Math.random() * 42,
                height: 30 + Math.random() * 10,
                alpha: 0.08 + Math.random() * 0.05
            });
        }
    }

    function laneY(laneIndex, x, time) {
        const lane = lanes[laneIndex];
        return lane.y + Math.sin((x * 0.0055) + time * 0.0012 + lane.phase) * lane.amplitude;
    }

    function drawGrid(time) {
        ctx.save();
        ctx.strokeStyle = 'rgba(75, 140, 255, 0.08)';
        ctx.lineWidth = 1;
        for (let x = -40; x < width + 40; x += 48) {
            ctx.beginPath();
            ctx.moveTo(x + Math.sin(time * 0.0003 + x * 0.01) * 6, 0);
            ctx.lineTo(x + Math.sin(time * 0.0003 + x * 0.01) * 6, height);
            ctx.stroke();
        }
        for (let y = -40; y < height + 40; y += 48) {
            ctx.beginPath();
            ctx.moveTo(0, y + Math.cos(time * 0.00025 + y * 0.01) * 6);
            ctx.lineTo(width, y + Math.cos(time * 0.00025 + y * 0.01) * 6);
            ctx.stroke();
        }
        ctx.restore();
    }

    function drawProgrammingLayer(time) {
        ctx.save();
        circuitNodes.forEach(function (node, index) {
            const pulse = (Math.sin(time * 0.002 + node.pulse) + 1) * 0.5;
            const nodeY = node.y + Math.sin(time * 0.0015 + index) * 6;
            ctx.strokeStyle = 'rgba(124, 231, 255, 0.08)';
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.moveTo(node.x - node.width, nodeY);
            ctx.lineTo(node.x - 18, nodeY);
            ctx.lineTo(node.x - 6, nodeY - 12);
            ctx.lineTo(node.x + 10, nodeY - 12);
            ctx.lineTo(node.x + 24, nodeY);
            ctx.lineTo(node.x + node.width, nodeY);
            ctx.stroke();
            ctx.beginPath();
            ctx.fillStyle = 'rgba(124, 231, 255, ' + (0.16 + pulse * 0.24) + ')';
            ctx.shadowColor = 'rgba(124, 231, 255, 0.25)';
            ctx.shadowBlur = 12;
            ctx.arc(node.x, nodeY, 2.4 + pulse * 2.2, 0, Math.PI * 2);
            ctx.fill();
            ctx.shadowBlur = 0;
        });

        terminalBlocks.forEach(function (block, index) {
            const wobble = Math.sin(time * 0.0014 + index * 0.9) * 4;
            const x = block.x + wobble;
            const y = block.y;
            ctx.fillStyle = 'rgba(7, 15, 28, ' + block.alpha + ')';
            ctx.strokeStyle = 'rgba(124, 231, 255, 0.08)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.roundRect(x, y, block.width, block.height, 9);
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = 'rgba(124, 231, 255, 0.2)';
            ctx.beginPath();
            ctx.arc(x + 12, y + 10, 2.2, 0, Math.PI * 2);
            ctx.arc(x + 20, y + 10, 2.2, 0, Math.PI * 2);
            ctx.arc(x + 28, y + 10, 2.2, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = 'rgba(216, 231, 255, 0.18)';
            ctx.font = '11px "Plus Jakarta Sans", monospace';
            ctx.fillText('> run skill_' + (index + 1), x + 11, y + 23);
        });

        codeSymbols.forEach(function (symbol) {
            const y = symbol.y + Math.sin(time * 0.0013 * symbol.drift + symbol.phase) * 7;
            ctx.fillStyle = 'rgba(124, 231, 255, ' + symbol.alpha + ')';
            ctx.font = symbol.size + 'px "Plus Jakarta Sans", monospace';
            ctx.fillText(symbol.text, symbol.x, y);
        });
        ctx.restore();
    }

    function drawScore() {
        ctx.save();
        ctx.font = '800 18px "Plus Jakarta Sans", sans-serif';
        ctx.textAlign = 'right';
        ctx.fillStyle = '#ffd166';
        ctx.shadowColor = 'rgba(255, 209, 102, 0.6)';
        ctx.shadowBlur = 10;
        const scrollY = window.scrollY || document.documentElement.scrollTop;
        const posY = 120 - scrollY;
        if (posY > -30) {
            ctx.fillText('SCORE: ' + bgScore, width - 40, posY);
        }
        ctx.restore();
    }

    function drawLanes(time) {
        lanes.forEach(function (lane, index) {
            ctx.beginPath();
            ctx.lineWidth = index === pacman.lane ? 2.4 : 1.2;
            ctx.strokeStyle = index === pacman.lane ? 'rgba(52, 211, 255, 0.22)' : 'rgba(52, 211, 255, 0.1)';
            for (let x = -80; x <= width + 80; x += 8) {
                const y = laneY(index, x, time);
                if (x === -80) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
        });
    }

    function drawPellets(time, delta) {
        pellets.forEach(function (pellet) {
            pellet.x -= lanes[pellet.lane].speed * delta;
            
            if (pellet.x < -200) {
                pellet.x = width + 200 + Math.random() * 150;
                pellet.eaten = false;
                pellet.respawnTimer = 0;
            }

            if (pellet.eaten) {
                pellet.respawnTimer += delta;
                if (pellet.respawnTimer > 6 && Math.abs(pellet.x - pacman.x) > 400) {
                    pellet.eaten = false;
                    pellet.respawnTimer = 0;
                } else {
                    return;
                }
            }

            const y = laneY(pellet.lane, pellet.x, time) + Math.sin(time * 0.003 + pellet.offset) * 1.5;
            const glow = pellet.radius > 5 ? 18 : 8;

            ctx.beginPath();
            ctx.fillStyle = pellet.radius > 5 ? '#ffd166' : 'rgba(139, 239, 255, ' + pellet.alpha + ')';
            ctx.shadowColor = pellet.radius > 5 ? 'rgba(255, 209, 102, 0.8)' : 'rgba(139, 239, 255, 0.4)';
            ctx.shadowBlur = glow;
            ctx.arc(pellet.x, y, pellet.radius, 0, Math.PI * 2);
            ctx.fill();
            ctx.shadowBlur = 0;
        });
    }

    function spawnSpark(x, y) {
        sparks.push({
            x, y, life: 1, radius: 2 + Math.random() * 2,
            vx: -0.8 - Math.random() * 1.2,
            vy: (Math.random() - 0.5) * 1.2
        });
    }

    function drawPacman(time, delta) {
        const isPowered = powerModeTimer > 0;
        const effectiveSpeed = isPowered ? pacman.speed * 1.5 : pacman.speed;
        const effectiveRadius = isPowered ? pacman.radius * 1.5 : pacman.radius;

        if (pacman.dying) {
            pacman.deathTimer += delta;
            ctx.save();
            ctx.translate(pacman.x, pacman.y);
            if (pacman.direction === -1) ctx.scale(-1, 1);
            ctx.fillStyle = '#ffd166';
            ctx.shadowColor = 'rgba(255, 209, 102, 0.55)';
            ctx.shadowBlur = 24;
            const deathAngle = Math.min(Math.PI, pacman.deathTimer * 3.5);
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.arc(0, 0, pacman.radius, deathAngle, Math.PI * 2 - deathAngle);
            ctx.closePath();
            ctx.fill();
            ctx.restore();

            if (pacman.deathTimer > 1.2) {
                pacman.dying = false;
                pacman.deathTimer = 0;
                pacman.x = -100;
                let newLane;
                do { newLane = Math.floor(Math.random() * lanes.length); } while (newLane === pacman.lane);
                pacman.lane = newLane;
                pacman.direction = 1;
            }
            return;
        }

        pacman.x += (effectiveSpeed * pacman.direction) * delta;

        if (pacman.direction === 1 && pacman.x - effectiveRadius > width + 100) {
            pacman.x = -100;
            let newLane;
            do { newLane = Math.floor(Math.random() * lanes.length); } while (newLane === pacman.lane);
            pacman.lane = newLane;
        } else if (pacman.direction === -1 && pacman.x + effectiveRadius < -100) {
            pacman.x = width + 100;
            let newLane;
            do { newLane = Math.floor(Math.random() * lanes.length); } while (newLane === pacman.lane);
            pacman.lane = newLane;
        }

        pacman.y = laneY(pacman.lane, pacman.x, time);
        pacman.mouth = 0.22 + (Math.sin(time * 0.02 * pacman.mouthSpeed) + 1) * 0.17;

        pellets.forEach(p => {
            if (!p.eaten && p.lane === pacman.lane) {
                const d = Math.abs(p.x - pacman.x);
                if (d < (effectiveRadius + 6)) {
                    p.eaten = true;
                    bgScore += (p.radius > 5 ? 50 : 10);
                    if (p.radius > 5) powerModeTimer = 10;
                }
            }
        });

        ghosts.forEach(g => {
            if (!g.eaten && g.lane === pacman.lane) {
                const d = Math.abs(g.x - pacman.x);
                if (!isPowered) {
                    if (d < (effectiveRadius)) {
                        pacman.dying = true;
                        pacman.deathTimer = 0;
                    } else {
                        if (pacman.direction === 1 && g.x > pacman.x && d < 180) pacman.direction = -1;
                        else if (pacman.direction === -1 && g.x < pacman.x && d < 180) pacman.direction = 1;
                    }
                } else {
                    if (d < (effectiveRadius + 12)) {
                        g.eaten = true;
                        bgScore += 200;
                        setTimeout(() => {
                            g.eaten = false;
                            g.lane = Math.floor(Math.random() * lanes.length);
                            g.x = pacman.x > width / 2 ? -200 : width + 200;
                        }, 1000);
                    }
                }
            }
        });

        if (Math.random() > 0.72) {
            spawnSpark(pacman.x - (effectiveRadius * 0.5 * pacman.direction), pacman.y);
        }

        ctx.save();
        ctx.translate(pacman.x, pacman.y);
        const currentScale = isPowered ? 1.5 : 1;
        if (pacman.direction === -1) ctx.scale(-currentScale, currentScale);
        else ctx.scale(currentScale, currentScale);
        
        ctx.fillStyle = '#ffd166';
        ctx.shadowColor = 'rgba(255, 209, 102, 0.55)';
        ctx.shadowBlur = 24;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.arc(0, 0, pacman.radius, pacman.mouth, Math.PI * 2 - pacman.mouth);
        ctx.closePath();
        ctx.fill();
        ctx.fillStyle = '#0b1020';
        ctx.beginPath();
        ctx.arc(4, -8, 2.3, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
    }

    function drawGhost(ghost, time, delta) {
        if (ghost.eaten) return;

        const isFleeing = powerModeTimer > 0;
        const inSameLane = (ghost.lane === pacman.lane);
        
        if (isFleeing) {
            if (inSameLane) { ghost.direction = (ghost.x > pacman.x) ? 1 : -1; }
            ghost.x += (ghost.speed * 1.6 * ghost.direction) * delta;
        } else {
            if (inSameLane && !pacman.dying) {
                ghost.direction = (ghost.x > pacman.x) ? -1 : 1;
                ghost.x += (pacman.speed * 1.15 * ghost.direction) * delta; // 115% da velocidade base do Pac-Man
            } else {
                if (ghost.x < -200) {
                    ghost.x = width + 200 + Math.random() * 300;
                    ghost.lane = Math.floor(Math.random() * lanes.length);
                    ghost.direction = -1;
                } else if (ghost.x > width + 200) {
                    ghost.x = -200;
                    ghost.lane = Math.floor(Math.random() * lanes.length);
                    ghost.direction = 1;
                }
                ghost.x += (ghost.speed * ghost.direction) * delta;
            }
        }

        ghost.y = laneY(ghost.lane, ghost.x, time) + Math.sin(time * 0.004 + ghost.bob) * 4;

        ctx.save();
        ctx.translate(ghost.x, ghost.y);
        ctx.fillStyle = isFleeing ? '#000044' : ghost.color.body;
        ctx.shadowColor = isFleeing ? '#0055ff' : ghost.color.body;
        ctx.shadowBlur = 18;
        ctx.beginPath();
        ctx.moveTo(-ghost.width / 2, ghost.height / 2);
        ctx.lineTo(-ghost.width / 2, 0);
        ctx.arc(0, 0, ghost.width / 2, Math.PI, 0);
        ctx.lineTo(ghost.width / 2, ghost.height / 2);
        for (let i = 0; i < 4; i++) {
            const waveX = ghost.width / 2 - i * (ghost.width / 3);
            const waveY = ghost.height / 2 + (i % 2 === 0 ? 4 : -2);
            ctx.quadraticCurveTo(waveX - 4, waveY, waveX - 8, ghost.height / 2);
        }
        ctx.closePath();
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.fillStyle = isFleeing ? '#4d79ff' : ghost.color.eye;
        ctx.beginPath();
        ctx.arc(-6, -2, 4.5, 0, Math.PI * 2);
        ctx.arc(6, -2, 4.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = isFleeing ? '#fff' : '#14213d';
        ctx.beginPath();
        ctx.arc(-5 + Math.sin(time * 0.003) * 1.2, -1, 2.1, 0, Math.PI * 2);
        ctx.arc(7 + Math.sin(time * 0.003) * 1.2, -1, 2.1, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
    }

    function drawSparks(delta) {
        sparks = sparks.filter(function (spark) {
            spark.x += spark.vx * 60 * delta;
            spark.y += spark.vy * 60 * delta;
            spark.life -= 0.03;
            if (spark.life <= 0) return false;
            ctx.beginPath();
            ctx.fillStyle = 'rgba(255, 209, 102, ' + spark.life + ')';
            ctx.arc(spark.x, spark.y, spark.radius * spark.life, 0, Math.PI * 2);
            ctx.fill();
            return true;
        });
    }

    function render(time) {
        if (!lastFrame) lastFrame = time;
        const delta = Math.min((time - lastFrame) / 1000, 0.033);
        lastFrame = time;
        if (powerModeTimer > 0) powerModeTimer -= delta;
        ctx.clearRect(0, 0, width, height);
        drawGrid(time);
        drawProgrammingLayer(time);
        drawLanes(time);
        drawPellets(time, delta);
        drawPacman(time, delta);
        ghosts.forEach(ghost => drawGhost(ghost, time, delta));
        drawSparks(delta);
        drawScore();
        if (!motionQuery.matches) animationId = window.requestAnimationFrame(render);
    }

    function renderStatic(time) {
        ctx.clearRect(0, 0, width, height);
        drawGrid(time);
        drawProgrammingLayer(time);
        drawLanes(time);
        drawPellets(time, 0.016);
        drawPacman(time, 0.016);
        ghosts.forEach(ghost => drawGhost(ghost, time, 0.016));
        drawScore();
    }

    function handleMotionChange() {
        window.cancelAnimationFrame(animationId);
        lastFrame = 0;
        if (motionQuery.matches) renderStatic(performance.now());
        else animationId = window.requestAnimationFrame(render);
    }

    resizeCanvas();
    handleMotionChange();
    window.addEventListener('resize', resizeCanvas);
    if (typeof motionQuery.addEventListener === 'function') {
        motionQuery.addEventListener('change', handleMotionChange);
    } else if (typeof motionQuery.addListener === 'function') {
        motionQuery.addListener(handleMotionChange);
    }
});
