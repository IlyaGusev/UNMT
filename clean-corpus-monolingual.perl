#!/usr/bin/env perl
#
# This file is part of moses.  Its use is licensed under the GNU Lesser General
# Public License version 2.1 or, at your option, any later version.

# $Id: clean-corpus-n.perl 3633 2010-10-21 09:49:27Z phkoehn $
use warnings;
use strict;
use Getopt::Long;
my $help;
my $lc = 0; # lowercase the corpus?
my $ignore_ratio = 0;
my $ignore_xml = 0;
my $enc = "utf8"; # encoding of the input and output files
    # set to anything else you wish, but I have not tested it yet
my $max_word_length = 1000; # any segment with a word (or factor) exceeding this length in chars
    # is discarded; motivated by symal.cpp, which has its own such parameter (hardcoded to 1000)
    # and crashes if it encounters a word that exceeds it
my $ratio = 9;

GetOptions(
  "help" => \$help,
  "lowercase|lc" => \$lc,
  "encoding=s" => \$enc,
  "ratio=f" => \$ratio,
  "ignore-ratio" => \$ignore_ratio,
  "ignore-xml" => \$ignore_xml,
  "max-word-length|mwl=s" => \$max_word_length
) or exit(1);

if (scalar(@ARGV) < 4 || $help) {
    print "syntax: clean-corpus-n.perl [-ratio n] corpus clean-corpus [lines retained file]\n";
    exit;
}

my $corpus = $ARGV[0];
my $out = $ARGV[1];
my $min = $ARGV[2];
my $max = $ARGV[3];

my $linesRetainedFile = "";
if (scalar(@ARGV) > 6) {
	$linesRetainedFile = $ARGV[6];
	open(LINES_RETAINED,">$linesRetainedFile") or die "Can't write $linesRetainedFile";
}

sub clean_file {
  my ($opn, $out) = @_;
  my $innr = 0;
  my $outnr = 0;
  my $factored_flag;

  open(F,$opn) or die "Can't open '$opn'";
  open(FO,">$out") or die "Can't write $out";

  # necessary for proper lowercasing
  my $binmode;
  if ($enc eq "utf8") {
    $binmode = ":utf8";
  } else {
    $binmode = ":encoding($enc)";
  }
  binmode(F, $binmode);
  binmode(FO, $binmode);
  while(my $f = <F>) {
    $innr++;
    print STDERR "." if $innr % 10000 == 0;
    print STDERR "($innr)" if $innr % 100000 == 0;
    chomp($f);
    if ($innr == 1) {
      $factored_flag = ($f =~ /\|/);
    }

    #if lowercasing, lowercase
    if ($lc) {
      $f = lc($f);
    }

    $f =~ s/\|//g unless $factored_flag;
    $f =~ s/\s+/ /g;
    $f =~ s/^ //;
    $f =~ s/ $//;
    next if $f eq '';

    my $fc = &word_count($f);
    next if $fc > $max;
    next if $fc < $min;
    # Skip this segment if any factor is longer than $max_word_length
    my $max_word_length_plus_one = $max_word_length + 1;
    next if $f =~ /[^\s\|]{$max_word_length_plus_one}/;

    # An extra check: none of the factors can be blank!
    die "There is a blank factor in $corpus on line $innr: $f"
      if $f =~ /[ \|]\|/;

    $outnr++;
    print FO $f."\n";

    if ($linesRetainedFile ne "") {
  	print LINES_RETAINED $innr."\n";
    }
  }
}

my $opn = undef;
my $l1input = "$corpus";
if (-e $l1input) {
  $opn = $l1input;
} elsif (-e $l1input.".gz") {
  $opn = "gunzip -c $l1input.gz |";
} else {
    die "Error: $l1input does not exist";
}
clean_file($opn, "$out");

if ($linesRetainedFile ne "") {
  close LINES_RETAINED;
}

print STDERR "\n";

sub word_count {
  my ($line) = @_;
  if ($ignore_xml) {
    $line =~ s/<\S[^>]*\S>/ /g;
    $line =~ s/\s+/ /g;
    $line =~ s/^ //g;
    $line =~ s/ $//g;
  }
  my @w = split(/ /,$line);
  return scalar @w;
}
