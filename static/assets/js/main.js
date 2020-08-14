function validate() {  
    if (confirm("Do you want to classify the images?")) {
        return true;
    }
    else {
        return false; 
    }            
}

        
function handleResponse(response)
{
   if (response == "")
	  $('#resultsParagraph').html("(None)");
   else
	  $('#resultsParagraph').html(response);
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
			success: handleResponse
		} );
}
/*
$("button").click(function(e) {
    e.preventDefault();

}); */
         

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