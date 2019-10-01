var lineFit = (function() {

    var exports = {};

////////////////////////////////// global variables

    //d3 chart components
    var  chart;

    var xMin = -20;
    var xMax = 20;
    var yMin = -10;
    var yMax = 10;

    var outer_height = 300;
    var outer_width = 500;

    var margin = { top: 20, right: 20, bottom: 20, left: 20 }
    var chart_width = outer_width - margin.left - margin.right;
    var chart_height = outer_height -margin.top - margin.bottom;

    var x_scale = d3.scale.linear().domain([xMin,xMax]).range([0,chart_width]);
    var y_scale = d3.scale.linear().domain([yMin,yMax]).range([chart_height,0]);
    var x_scale2 = d3.scale.linear().domain([0,chart_width]).range([xMin,xMax]);
    var y_scale2 = d3.scale.linear().domain([chart_height,0]).range([yMin,yMax]);

////////////////////////////////// helper functions

    //rounds a number (number) to the specified amount of decimal points (decimals)
    function round_number(number,decimals){
        return Math.round(number*Math.pow(10,decimals))/Math.pow(10,decimals)
    }

    //creates a range of values from start to stop in step sized increments
    function range(start, stop, step){
        if (typeof stop=='undefined'){
            // one param defined
            stop = start;
            start = 0;
        };
        if (typeof step=='undefined'){
            step = 1;
        };
        if ((step>0 && start>=stop) || (step<0 && start<=stop)){
            return [];
        };
        var result = [];
        for (var i=start; step>0 ? i<stop : i>stop; i+=step){
            result.push(i);
        };
        return result;
    };

    // sigmoid function
    function sigmoid(z) {
      let g = math.eval(`1 ./ (1 + e.^-z)`, {
        z,
      });

      return g;
    }

    function sigmoid_2(z,coefficients){
      let p = coefficients[1]*z + coefficients[0];
      let g = math.eval(`1 ./ (1 + e.^-p)`, {
        p,
      });

      return g;
    }

/////////////////////////////////// set up div functions

    /* Model that contains instance variables:
        pointList - points to be fitted, one array of many [x,y] arrays
        currentCoeffs - [a,b] of y = ax + b

        functions:
        add_point - adds a point to pointList
        replace_point - takes an index and new x and y coordinates and replaces the point at the index with the new coordinates
        clear_points - empties the pointList
        getIndexOf - takes x and y coordinates and gets the index of the point in the pointList
        get_point_list - returns the pointList
        change_line - takes new coefficients to change currentCoeffs to
        change_a - changes the first entry in currentCoeffs to the input
        change_b - changes the second entry in currentCoeffs to the input
        get_a - returns the a value
        get_b - returns the b value
        getCoeffs - return current line coefficients
        randomize_points - generate a number of random points to add to pointList
        make_random_point - makes a random point between xMax,xMin and yMax,yMin
        findError - returns the error between a point and the line
    */
    function Model() {
        var pointList = []; //array of [x,y] arrays
        var currentCoeffs = [0,0]; //[a,b] where a and b are from y = ax + b

        function add_point(point){ // add a point
            pointList.push([point[0],point[1]]);
        }

        function replace_point(index,x,y){
            delete pointList[index];
            pointList[index] = [x,y];
        }

        function getIndexOf(x,y){
            for (var i = 0; i < pointList.length; i++) {
                if(pointList[i][0] == x && pointList[i][1] == y)
                    return i;
            };

            return -1;
        };

        function get_point_list(){
            return pointList;
        }
        function change_line(newCoeffs){ //change the coefficients of the best fit line
            currentCoeffs = newCoeffs;
        }
        function change_a(a){
            currentCoeffs[1] = a;
        }
        function change_b(b){
            currentCoeffs[0] = b;
        }
        function get_a(){
            return currentCoeffs[1];
        }
        function get_b(){
            return currentCoeffs[0];
        }

        function getCoeffs(){ //return the coefficients of the current best fit line
            return currentCoeffs;
        }

        function randomize_points(number){
            pointList = [];
            yMin = 0;
            yMax = 1;
            xMax = 10;
            xMin = -10;
            for(var i=0; i<number; i++){
                pointList.push(make_random_point());
            }
            return pointList;
        }

        function make_random_point(){
            var x = Math.random()*xMax;
            var isNeg = Math.random();
            if(isNeg>0.5){
                x = (-1)*x;
            }
            var y = Math.random()*yMax;
            if(y>=0.5){
              y = 1;
            }
            else{
              y = 0;
            }
            return [round_number(x,2),round_number(y,2)]
        }

        //finds the vertical error between a point and the line
        function findError(point){
            var error = point[1]-lineAt(point[0]);
            return error;
        }


        //returns the y value of the line at a point
        function lineAt(x){
            return sigmoid((currentCoeffs[1]*x)+currentCoeffs[0]);
        }

        //sums the squared vertical error from each point to the line
        function sumOfSquares(){
            var sumOfSquareError = 0;
            for(var i=0; i<pointList.length; i++){
                sumOfSquareError += Math.pow(findError(pointList[i]),2);
            }
            return sumOfSquareError;
        }

        // finds the error of the logistic model
        function costFunctionLog(){

          const m = pointList.length

          let total_cost = 0
          for (var i=0; i<pointList.length; i++){
            let h = sigmoid((currentCoeffs[1]*pointList[i][0])+currentCoeffs[0]);
            let cost = ((-pointList[i][1] * Math.log(h)) - (1-pointList[i][1])*Math.log(1-h));
            total_cost += cost;
          }
          console.log("cost:" + (1/m) * total_cost)
          return (1/m) * total_cost;
        }

        //returns the array of points and their squared error
        function points_with_square_error(){
            var new_list = [];
            for(var i=0; i<pointList.length; i++){
                new_list.push([{y: Math.pow(findError(pointList[i]),2)}])
            }

            return new_list;
        }

        //returns the array of points and their absolute error
        function points_with_abs_error(){
            var new_list = [];
            for(var i=0; i<pointList.length; i++){
                new_list.push([{y: findError(pointList[i])}])
            }
            return new_list;
        }

        function point_log_loss(){
            const m = pointList.length
            var new_list = [];
            for(var i=0; i<pointList.length; i++){
                let h = sigmoid((currentCoeffs[1]*pointList[i][0])+currentCoeffs[0]);
                let cost = (1/m) * ((-pointList[i][1] * Math.log(h)) - (1-pointList[i][1])*Math.log(1-h));
                new_list.push([{y: cost}])
            }
          return new_list;
        }

        //finds the best fit for the points on the graph
        function bestFit(){
            var lineCoeffs; //coefficients of y=sigmoid(ax+b) in the form [a,b]

            lineCoeffs = logistic_regression();
            return lineCoeffs;
        }

        // Logistic regression algorithm adapted from Robin Wieruch
        function logistic_regression()
        {

          // sigmoid function
          function sigmoid(z) {
            let g = math.eval(`1 ./ (1 + e.^-z)`, {
              z,
            });

            return g;
          }

          // cost function (log-loss)
          function costFunction(theta, X, y) {

            const m = y.size()[0];

            let h = sigmoid(math.eval(`X * theta`, {
              X,
              theta,
            }));

            const cost = math.eval(`(1 / m) * (-y' * log(h) - (1 - y)' * log(1 - h))`, {
              h,
              y,
              m,
            });

            const grad = math.eval(`(1 / m) * (h - y)' * X`, {
              h,
              y,
              m,
              X,
            });

            return { cost, grad };
          }

          // gradient descent function
          function gradientDescent(X, y, theta, ALPHA, ITERATIONS) {

            const m = y.size()[0];

            for (let i = 0; i < ITERATIONS; i++) {
              let h = sigmoid(math.eval(`X * theta`, {
                X,
                theta,
              }));

              theta = math.eval(`theta - ALPHA / m * ((h - y)' * X)'`, {
                theta,
                ALPHA,
                m,
                X,
                y,
                h,
              });
            }
            return theta;
          }

          var X_array = [];
          var y_array = [];
          // grab x values and y values from pointList
          for(i=0;i<pointList.length;i++)
          {
              // this is our data pair
              x_val = pointList[i][0]; y_val = pointList[i][1];

              X_array.push([x_val]);
              y_array.push([y_val]);
          }
          let X = math.matrix(X_array);
          let y = math.matrix(y_array);

          let m = y.size()[0];
          let n = 1;

          // Add Intercept Term
          X = math.concat(math.ones([m, 1]).valueOf(), X);

          // initialize theta to 0
          let theta = Array(n + 1).fill().map(() => [0]);
          let { cost: untrainedThetaCost, grad } = costFunction(theta, X, y);

          // set alpha value (learning rate) and number of iterations
          const ALPHA = 0.001;
          const ITERATIONS = 400;

          theta = [[0], [0]];
          theta = gradientDescent(X, y, theta, ALPHA, ITERATIONS);

          const { cost: trainedThetaCost } = costFunction(theta, X, y)
          return [theta['_data'][0][0],theta['_data'][1][0]] // trainedThetaCost

        }

        //sums the errors of the points and returns optimized a and b for y = ax + b
        function linear_regression()
        {
            var i, x, y,
                sumx=0, sumy=0, sumx2=0, sumy2=0, sumxy=0,
                a, b;
            var count = pointList.length;

            for(i=0;i<pointList.length;i++)
            {
                // this is our data pair
                x_val = pointList[i][0]; y_val = pointList[i][1];

                X.push(x_val)
                y.push(y_val)
            }

            // note: the denominator is the variance of the random variable X
            // the only case when it is 0 is the degenerate case X==constant
            var b = (sumy*sumx2 - sumx*sumxy)/(count*sumx2-sumx*sumx);
            var a = (count*sumxy - sumx*sumy)/(count*sumx2-sumx*sumx);

            return [a,b];
        }

        //finds the statistical variance of the points
        function get_variance(){
            var n = pointList.length;
            if(n ==0){
                return 0;
            }
            var sum = 0;
            for(var i =0; i<n; i++){
                sum += pointList[i][0];
            }
            var mean = sum/n;
            var variance = 0;
            for(var i =0; i<n; i++){
                variance += Math.pow(pointList[i][0]-mean,2);
            }
            return variance;
        }

        return {add_point: add_point, get_point_list: get_point_list, change_line: change_line, getCoeffs: getCoeffs, change_a: change_a, get_a: get_a, change_b: change_b, get_b: get_b, findError: findError, lineAt: lineAt, bestFit: bestFit, linear_regression: logistic_regression, costFunctionLog: costFunctionLog, get_variance: get_variance, points_with_square_error: points_with_square_error, getIndexOf: getIndexOf, points_with_abs_error: points_with_abs_error, point_log_loss: point_log_loss, randomize_points: randomize_points, replace_point: replace_point};
    }

    /* View that controls how the content is displayed to the user.
        contains instance variables:
        color_scale - a d3 object that converts numbers into colors

        functions:
        setupLineControls
        setupZeroDegreeControls
        displayLine
        updatePointsOnGraph
        displayErrorInfo
        removeErrorInfo
        updateEquation
        turnErrorDisplayOn
        turnErrorDisplayOff
        graph
        updateBestFitDisplay
        updateDisplay
    */
    function View(div,model) {
        var color_scale = d3.scale.linear()
                .domain([0, 1])
                .range(['#61A72D','#CC0000']);

        var tooltip = d3.select("body").append("div").attr("class","point-error").text("");

        var aSlider,bSlider,cSlider;
        //initialize the display as dealing with just lines

        aSlider = $(".a-slider").slider({ min: -10, max: 10, step: .01, slide: function( event, ui ) {
            if ($('.plot-fit').prop('checked')==true){
                $('.plot-fit').attr('checked', false);
            }
            model.change_a(ui.value);
            $('.a-label').html(ui.value);
            updateDisplay();
            }

        });

        bSlider = $(".b-slider").slider({ min: 1.5*yMin, max: 1.5*yMax, step: .01,
            slide: function( event, ui ) {
                if ($('.plot-fit').prop('checked')==true){
                    $('.plot-fit').attr('checked', false);
                }
                model.change_b(ui.value);
                $('.b-label').html(ui.value);
                updateDisplay();
            },
        });

        setupLineControls();
        setupButtons();

        setupGraph(-10,10,0,1);
        displayLine([0,0],false);

        //controls for when the user wants to plot a first-order line
        function setupLineControls(){
            $('.a-header').show();
            $('.a-label').show();
            aSlider.css("width", "80%");
            aSlider.show();//.slider( "enable" );
            $('.b-header').show();
            $('.b-label').show();
            bSlider.css("width", "80%");
            bSlider.show();

            aSlider = $(".a-slider").slider({ min: -10, max: 10, step: .01, slide: function( event, ui ) {
                if ($('.plot-fit').prop('checked')==true){
                    $('.plot-fit').attr('checked', false);
                }
                model.change_a(ui.value);
                $('.a-label').html(ui.value);
                updateDisplay();
                }

            });

            aSlider.slider('option','value',model.get_a());
            bSlider.slider('option','value',model.get_b());
            $('.b-label').html(round_number(model.get_b(),2));
            $('.a-label').html(round_number(model.get_a(),2));

            if($('.plot-fit').prop("checked")){
                updateBestFitDisplay(true);
                turnErrorDisplayOn(false);
            }
            if(chart !== undefined){
                updateDisplay();
            }
        }

        //sets up the buttons
        function setupButtons(){

            $('.plot-fit').on("click",function(){
                updateDisplay()

            });

            $('.randomize').on("click",function(){
                $(".alert").remove();
                model.randomize_points($(".point-number").val());
                setupGraph(xMin,xMax,yMin,yMax);
                updateDisplay()
            });
        }

        function make_sigmoid_func(coef,intercept){
          var pow = Math.pow, e = Math.E;
          return (function(xi) {
            return 1/(1+pow(e,-(xi*coef + intercept)))
          });
        }

         //takes coefficients to y=ax+b and displays the corresponding on the graph
        function displayLine(coefficients,animate){

            if(!animate){
                chart.selectAll(".best-fit").data(range(xMin,xMax,0.1)).remove();
                chart.selectAll(".best-fit").data(coefficients).remove();

                sig_vals = []
                for (let i = -10; i <= 10; i++){
                  sig_vals.push([i,sigmoid_2(i,coefficients)])
                }

                var lineFunction = d3.svg.line()
                                         .x(function(d){return x_scale(d[0])})
                                         .y(function(d){return y_scale(d[1])})
                                         .interpolate("basis");

                chart.selectAll(".best-fit").data(coefficients).enter().append("path").attr("class", "best-fit").attr("d",lineFunction(sig_vals)).attr("fill", "none");

                turnErrorDisplayOff();
                turnErrorDisplayOn(false);
            }
            else{

                var y1 = coefficients[0]*xMin+coefficients[1];
                var y2 = coefficients[0]*xMax+coefficients[1];

                if(chart.selectAll(".best-fit")[0].length> 0){
                    chart.selectAll(".best-fit").transition().duration(750).attr('x1', x_scale(xMin)).attr('x2', x_scale(xMax)).attr('y1', y_scale(y1)).attr('y2',y_scale(y2));
                }
                else{
                     chart.selectAll(".best-fit").data(coefficients).enter().append("line").attr("class", "best-fit").attr('x1', x_scale(xMin)).attr('x2', x_scale(xMax)).attr('y1', y_scale(y1)).attr('y2',y_scale(y2));

                }

                turnErrorDisplayOn(true);
            }
        }

        //plots all the points in the model's pointList to the svg
        function updatePointsOnGraph(){
            chart.selectAll(".datapoint").remove();
            var points = model.get_point_list();
            var point_index;
            chart.selectAll(".datapoint").data(points).enter().append("circle")
                .attr("class", "datapoint")
                .attr("cx", function(d){return x_scale(d[0])})
                .attr("cy", function(d){return y_scale(d[1])})
                .on("mouseover", function(d){
                    point_index = model.getIndexOf(d[0],d[1]);
                    $('.graphic > .translation > .layer:nth-of-type('+(point_index+1)+')').css("stroke","black");
                    $('.graphic > .translation > .layer:nth-of-type('+(point_index+1)+')').css("stroke","blue").css("stroke-width","3").css("stroke","5,3");
                })
                .on("mousemove", function(){
                    tooltip.style("top",(d3.event.pageY+10)+"px").style("left",(d3.event.pageX+10)+"px");
                })
                .on("mouseout",function(){
                    $('tr').find('#'+point_index).closest("tr").css("outline","none");
                    $('.graphic > .translation > .layer:nth-of-type('+(point_index+1)+')').css("stroke","none");
                    tooltip.style("visibility", "hidden");
                })
                .attr("id", function(d){
                    point_index = model.getIndexOf(d[0],d[1]);
                    return point_index;
                })
                .style("fill",'blue')
                .call(move)
                .attr("r", "4");
        }

        //shows the total error and sum of squares error
        function displayErrorInfo(){
            $(".info-container").empty();
            $(".info-container").append("<div class='row-fluid'><span class = 'squared'></span></div>");
            console.log("coefficients:" + model.getCoeffs())
            $(".squared").html("Log-Loss: " +round_number(model.costFunctionLog(),2));
        }

        //updates the displays equation to have the proper a, b
        function updateEquation(){
            var coefficients = model.getCoeffs();
            $('.equation').html("y = sigmoid(<span class='a-display' contenteditable = 'true'>"+round_number(coefficients[1],2)+"</span>x + <span class='b-display' contenteditable = 'true'>" + round_number(coefficients[0],2) + "</span>)");


            var contentsA = $('.a-display').html();
            $('.a-display').blur(function() {
                if (contentsA!=$(this).html()){
                    model.change_a(parseFloat($(this).html()));
                    contentsA = $(this).html();
                    aSlider.slider("option","value",model.get_a());
                    $('.a-label').html(round_number(model.get_a(),2));
                    updateDisplay();
                }
            });
            var contentsB = $('.b-display').html();
            $('.b-display').blur(function() {
                if (contentsB!=$(this).html()){
                    model.change_b(parseFloat($(this).html()));
                    contentsB = $(this).html();
                    bSlider.slider("option","value",model.get_b());
                    $('.b-label').html(round_number(model.get_b(),2));
                    updateDisplay();
                }
            });
        }

        function removeErrorInfo(){
            $(".info-container").empty();
            $(".squared").popover('disable');
        }

        //adds vertical bars from point to best-fit line (with color scale that displays how much error)
        function turnErrorDisplayOn(animate){
            if(!animate){
                chart.selectAll(".error-line").data(model.get_point_list()).enter().append("line").attr("class", "error-line").attr('x1', function(d){return x_scale(d[0])}).attr('x2', function(d){ return x_scale(d[0])}).attr('y1', function(d){ return y_scale(d[1]);}).attr('y2',function(d){ return y_scale(model.lineAt(d[0]));}).style("stroke", function(d) {return color_scale(model.findError(d)); });
            }
            else{
                chart.selectAll(".error-line").data(model.get_point_list()).transition().duration(750).attr('x1', function(d){return x_scale(d[0])}).attr('x2', function(d){ return x_scale(d[0])}).attr('y1', function(d){ return y_scale(d[1])}).attr('y2',function(d){ return y_scale(model.lineAt(d[0]))}).style("stroke", function(d) {return color_scale(model.findError(d)); });
            }

            displayErrorInfo()

        }

        var move =  d3.behavior.drag().on("drag",drag)

        function drag(){
            var dragPoint = d3.select(this);
            dragPoint
                .attr("cx",function(){return d3.event.dx + parseInt(dragPoint.attr("cx"));})
                .attr("cy",function(){return d3.event.dy +parseInt(dragPoint.attr("cy"));})
                var newX = x_scale2(parseInt(dragPoint.attr("cx")));
                var newY = y_scale2(parseInt(dragPoint.attr("cy")));
                model.replace_point(dragPoint.attr("id"), newX, newY);
                updateDisplay();
        }

        //removes vertical bars from point to best-fit line
        function turnErrorDisplayOff(){
            chart.selectAll(".error-line").remove();
            removeErrorInfo()
        }

        //displays the graph of sum of squared error, color coded to show which point contributes which block of error
        function graph(){
            $(".graph-container").empty();
            //var maxValue = model.get_variance()*5;
            var maxValue = 10;
            //var title = "Sum of Squares";
            var data = model.point_log_loss();

            var normal_error = model.points_with_abs_error();

            var graph_outer_width = 50;
            var graph_outer_height = 300;
            var graph_margin = { top: graph_outer_width/8, right: 30, bottom: graph_outer_width/8, left: 0 }
            var graph_chart_width = graph_outer_width - graph_margin.left - graph_margin.right;
            var graph_chart_height = graph_outer_height -graph_margin.top - graph_margin.bottom;

            var graph_y_scale = d3.scale.linear().domain([0,maxValue]).range([graph_chart_height,0]);

            var graph_chart = d3.select(".graph-container").append("svg").attr("class","graphic").attr("height", graph_outer_height).attr("width",graph_outer_width).append("g").attr("class","translation").attr("transform","translate(" + (graph_margin.left+graph_margin.right) + "," + (graph_margin.top + graph_margin.bottom -5)+ ")");

            graph_chart.selectAll(".y-scale-label").data(graph_y_scale.ticks(4)).enter().append("text").attr("class", "y-scale-label").attr("x",graph_margin.left/2).attr('y',graph_y_scale).attr("text-anchor","end").attr("dy","0.3em").attr("dx",-graph_margin.left/2).text(function(d){return d});

            if(data.length>0){
                var stack = d3.layout.stack();
                var stacked_data = stack(data);
                var layer_groups = graph_chart.selectAll(".layer").data(stacked_data).enter().append("g").attr("class", "layer");

                var rects = layer_groups.selectAll('rect').data(function(d){console.log(d); return d}).enter().append('rect').attr("x",0).style("fill", function(d, i, j) {return color_scale(data[j][0].y);}).attr("height", 0).attr("y", function(d){return graph_y_scale(d.y0)}).attr("y", function(d){return graph_y_scale(d.y0+d.y)}).attr("width", graph_chart_width).attr("height", function(d){ return graph_y_scale(d.y0) - graph_y_scale(d.y0+d.y); });
            }

      }

      function showCorrectSliders(){
        var coefficients = model.getCoeffs();
        aSlider.slider("option","value",coefficients[1]);
        $('.a-label').html(round_number(coefficients[1],2));
        bSlider.slider("option","value",coefficients[0]);
        $('.b-label').html(round_number(coefficients[0],2));
      }

        //plots the best fit line
        function updateBestFitDisplay(animate){
            var coeffs = model.bestFit()
            model.change_line(coeffs);
            updateEquation();
        }

        //updates the points, error bars, graph, equation
        function updateDisplay(){
            updatePointsOnGraph();
            if($('.plot-fit').prop("checked")){
                updateBestFitDisplay(true);
            }
            showCorrectSliders();
            displayLine(model.getCoeffs(),false);

            turnErrorDisplayOn(false);
            displayErrorInfo();
            updateEquation();
            graph();
        }

        return {displayLine: displayLine, displayErrorInfo: displayErrorInfo, updateBestFitDisplay: updateBestFitDisplay, updateEquation: updateEquation, updatePointsOnGraph: updatePointsOnGraph, updateDisplay: updateDisplay};
    }

    //set up svg with axes and labels
    function setupGraph(xMin,xMax,yMin,yMax){
        xMin = xMin;
        xMax = xMax;
        yMin = yMin;
        yMax = yMax;

        x_scale = d3.scale.linear().domain([xMin,xMax]).range([0,chart_width]);
        y_scale = d3.scale.linear().domain([yMin,yMax]).range([chart_height,0]);
        x_scale2 = d3.scale.linear().domain([0,chart_width]).range([xMin,xMax]);
        y_scale2 = d3.scale.linear().domain([chart_height,0]).range([yMin,yMax]);

        $(".chart-container").empty();
        chart = d3.select(".chart-container").append("svg").attr("class","chart").attr("height", outer_height).attr("width",outer_width).append("g").attr("transform","translate(" + margin.left + "," + margin.top + ")");

        chart.selectAll(".y-line").data(y_scale.ticks(10)).enter().append("line").attr("class", "y-line").attr('x1', 0).attr('x2', chart_width).attr('y1', y_scale).attr('y2',y_scale);

        chart.selectAll(".x-line").data(x_scale.ticks(10)).enter().append("line").attr("class", "x-line").attr('x1', x_scale).attr('x2', x_scale).attr('y1', 0).attr('y2',chart_height);

        chart.selectAll(".y-scale-label").data(y_scale.ticks(10)).enter().append("text").attr("class", "y-scale-label").attr("x",x_scale(0)).attr('y',y_scale).attr("text-anchor","end").attr("dy","0.3em").attr("dx","0.5em").text(String);

        chart.selectAll(".x-scale-label").data(x_scale.ticks(10)).enter().append("text").attr("class", "x-scale-label").attr("x",x_scale).attr('y',y_scale(0)).attr("text-anchor","end").attr("dy","0.3em").attr("dx","0.5em").text(String);

    }

    //setup main structure of app
    function setup(div) {

        var model = Model();
        var view = View(div, model);

        //initializes a nice little set of 4 points to begin with
        model.randomize_points(8);
        model.change_a(1)
        view.updateDisplay();
    };

    exports.setup = setup;
    exports.round_number = round_number;
    exports.model = Model;
    exports.view = View;

    return exports;
}());

$(document).ready(function() {
    lineFit.setup();
});
