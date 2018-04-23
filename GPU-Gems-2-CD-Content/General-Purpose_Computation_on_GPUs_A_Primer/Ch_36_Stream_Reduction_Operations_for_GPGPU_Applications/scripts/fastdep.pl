# fastdep.pl

my $obj_prefix = "";
my $obj_suffix = ( $^O eq "MSWin32" ) ? ".obj" : ".o";
my @extra_targets = ();
my $verbose = 0;

sub dir {

  my $file = shift;

  my $dir;
  if ( $file =~ /\// ) {
    $file =~ /^(.*)\/([^\/])+$/;
    $dir = $1;
  } else {
    $dir = ".";
  }

  $dir .= "/";
}

sub includes {

  my $file = shift;

  return @{$inc{$file}} if ( $inc{$file} );

  my ( @angles, @quotes );
  unless ( open(SCAN, $file) ) {
    print STDERR "fastdep: \"$file\": $!" if ( $verbose );
    return ();
  }

  while (<SCAN>) {


    next unless /^\s*\#/;
    if ( /^\s*\#\s*include\s*([<\"])(.*)[>\"]/ ) {
      if ( $1 eq "<" ) {
	push @angles, $2;
      } else {
	push @quotes, $2;
      }
    }
  }

  close(SCAN);

  my $dir = dir($file);
  my ( @files, $name );

  while ( $name = pop @quotes ) {
    my $f = $dir . $name;
    if ( -f $f ) {
      push @files, $f;
    } else {
      push @angles, $name;
    }
  }

  foreach $name ( @angles ) {
    my $found = 0;
    foreach $dir ( @incpath ) {
      $f = $dir . $name;
      if ( -f $f ) {
	push @files, $f;
	$found = 1;
	last;
      }
    }
    if ( !$found ) {
      print STDERR "fastdep: couldn't find \"$name\"\n" if ( $verbose );
    }
  }

  $inc{$file} = \@files;
  @files;
}

sub depends {

  my ( $file ) = @_;

  my ( @files ) = ( @_ );
  my ( %files );
  while ( $f = pop @files ) {

    next if exists $files{$f};
    $files{$f} = 1;

    push @files, includes($f);
  }

  keys %files;
}

foreach $arg ( @ARGV ) {

  if ( $arg =~ /^-I(.+)\/$/ ) {
    push @incpath, $1;
  }
  elsif ( $arg =~ /^-I(.+)$/ ) {
    push @incpath, $1 . "/";
  }
  elsif ( $arg =~ /^--obj-prefix=(.*)$/ ) {
    $obj_prefix = $1;
  }
  elsif ( $arg =~ /^--obj-suffix=(.*)$/ ) {
    $obj_suffix = $1;
  }
  elsif ( $arg =~ /^--extra-target=(.*)$/ ) {
    push @extra_targets, $1;
  }
  elsif ( $arg =~ /^--verbose$/ ) {
    $verbose = 1;
  }
  elsif ( $arg =~ /^-/ ) {
    # skip it
  }
  else {
    push @files, $arg;
  }
}

if ( $verbose ) {
  foreach $dir ( @incpath ) {
    if ( ! -d $dir ) {
      print STDERR "fastdep: \"$dir\": no such directory\n";
    }
  }
}

foreach $file ( @files ) {

  $file =~ /^(.*)\.\w+$/;
  my $obj  = $obj_prefix . $1 . $obj_suffix;
  foreach $t ( @extra_targets ) {
    $obj .= " " . $t;
  }
  foreach $deps ( depends($file) ) {
    print "$obj: $deps\n";
  }

  if ($file =~ /(.*)\.br$/) {
     $basefile = $1;

     print "$obj: $obj_prefix$basefile.cpp\n";
     print "$obj_prefix$basefile.cpp: \$(ROOTDIR)/bin/brcc\$(BINSUFFIX)\n";
     print "$obj_prefix$basefile.cpp: $file\n";
  }

  if ($file =~ /(.*)\.bri$/) {
     $basefile = $1;

     print "$obj: $obj_prefix$basefile.cpp\n";
     print "$obj_prefix$basefile.cpp: \$(ROOTDIR)/bin/brcc\$(BINSUFFIX)\n";
     print "$obj_prefix$basefile.cpp: $obj_prefix$basefile.br\n";
     print "$obj_prefix$basefile.br: $file\n";
  }
}
