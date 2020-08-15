// CONSTS
setTimeout(getResults, 1000);
var allUnzipped = Boolean(true);

// FUNCTIONS
function confirmClassification() {  
    if (confirm("Do you want to classify the images?")) {
		$("#loading").css("visibility", "visible");
		return true;
    }
    else {
        return false; 
    }            
}

function handleResponseInitialClassification(response)
{ 
	if (response == ""){
		$('#classification_msg').html('An error occured.');
	}
   else {
		$('#classification_msg').html(response);
		$("#loading").css("visibility", "visible");
	}
}
	
function validate() {
	if (confirmClassification() === true) {
			let url = '/classify'
		if (request != null)
			request.abort();
		request = $.ajax(
			{
				type: "GET",
				url: url,
				success: handleResponseInitialClassification
			}
		);
	}	
}

// probs can do more here
function validateUpload()
{
	allUnzipped = Boolean(true);
}

function handleResponseClassification(response)
{
	if ((response == 'Classification finished.') && ($("#classification_msg").html() === "Performing classification... ")){
		$("#loading").css("visibility", "hidden");
		$("#classification_msg").html('Classification has finished! Click "Prepare Visualization" to visualize the results')
		// set classificaiotn msg to nothing
		alert('Classification has finished! Click "Prepare Visualization" to visualize the results');
	}
}
        
function handleResponse(response)
{

	var prevhtmlString = $('#resultsParagraph').html()
	if (response == ""){
		$('#resultsParagraph').html("(No tiles found.)");
	}
   else {
	   $('#resultsParagraph').html(response);
	}
		   
	var htmlString = $('#resultsParagraph').html()

	if ((htmlString !== "(None)") && (htmlString !== "(No tiles found.)")) {
		if ((htmlString.length == prevhtmlString.length) && (allUnzipped==Boolean(true))) {
			alert('Tiles are unzipped and ready for classification! View the tile names in the textbox and click classify to proceed.')
			allUnzipped = Boolean(false);
		}
	} 	
}

let request = null;
         
function getResults()
{    
	// let author = $('#authorInput').val();
	//author = encodeURIComponent(author);
	// let url = '/searchresults?author=' + author;
	let url = '/searchresults';
	if (request != null)
		request.abort();
	request = $.ajax( {
			type: "GET",
			url: url,
			success: handleResponse, 

		} );
	setTimeout(getResults, 3000);
}
/*
$("button").click(function(e) {
    e.preventDefault();

}); */
classification_req = null
setTimeout(classification_finished, 20000);
function classification_finished()
{    
	let url = '/classificationfin';
	if (classification_req != null)
		classification_req.abort();
	classification_req = $.ajax( {
			type: "GET",
			url: url,
			success: handleResponseClassification, 
		} );
	setTimeout(classification_finished, 20000);
}

/*
	Helios by HTML5 UP
	html5up.net | @ajlkn
	Free for personal and commercial use under the CCA 3.0 license (html5up.net/license)
*/

(function($) {

	var	$window = $(window),
		$body = $('body'),
		settings = {
		};

	// Breakpoints.
		breakpoints({
			wide:      [ '1281px',  '1680px' ],
			normal:    [ '961px',   '1280px' ],
			narrow:    [ '841px',   '960px'  ],
			narrower:  [ '737px',   '840px'  ],
			mobile:    [ null,      '736px'  ]
		});

	// Play initial animations on page load.
		$window.on('load', function() {
			window.setTimeout(function() {
				$body.removeClass('is-preload');
			}, 100);
		});
/*
	// Dropdowns.
		$('#nav > ul').dropotron({
			mode: 'fade',
			speed: 350,
			noOpenerFade: true,
			alignment: 'center'
		});
 */
	// Scrolly.
		// $('.scrolly').scrolly();

	// Nav.

		// Button.
			$(
				'<div id="navButton">' +
					'<a href="#navPanel" class="toggle"></a>' +
				'</div>'
			)
				.appendTo($body);

		// Panel.
			$(
				'<div id="navPanel">' +
					'<nav>' +
						$('#nav').navList() +
					'</nav>' +
				'</div>'
			)
				.appendTo($body)
				.panel({
					delay: 500,
					hideOnClick: true,
					hideOnSwipe: true,
					resetScroll: true,
					resetForms: true,
					target: $body,
					visibleClass: 'navPanel-visible'
				});
})(jQuery);